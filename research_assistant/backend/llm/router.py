"""
router.py
---------
Routes PDF-grounded generations to the fine-tuned Qwen-1.5B endpoint (served
from Kaggle, see ``notebooks/serve_finetuned.ipynb``), with the API LLM as
fallback.

  • Health check: ``GET {FINETUNED_BASE_URL}/models`` (3 s timeout, cached 60 s).
  • Circuit breaker: after any failure the endpoint is skipped for 120 s.
  • Calls time out after 45 s and are not retried.

Used for the agent's final answer on turns that read uploaded PDFs, and by the
``summarize_papers`` tool.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx
import openai
from pydantic import BaseModel

from backend.config import Settings, get_settings
from backend.eval.guardrails import sanitize_text
from backend.ingestion import Chunk
from backend.runtime import timed

log = logging.getLogger(__name__)

FINETUNED_LABEL = "Fine-tuned Qwen-1.5B (Kaggle GPU)"
SYSTEM_PROMPT = "You are a research assistant. Cite sources as [Doc N, Sec S]."
HEALTH_TIMEOUT_S = 3.0
HEALTH_CACHE_S = 60.0
CIRCUIT_OPEN_S = 120.0
CALL_TIMEOUT_S = 45.0
ANSWER_MAX_TOKENS = 1024
ANSWER_TEMPERATURE = 0.2
MAX_PROMPT_CHUNKS = 4


class RoutedAnswer(BaseModel):
    """A final answer plus which model produced it."""

    answer: str
    model_used: str
    latency_ms: float | None = None
    fallback_reason: str | None = None


class FinetunedHealth(BaseModel):
    up: bool
    latency_ms: float | None = None
    circuit_open: bool


class Completion(BaseModel):
    """Result of :meth:`FinetunedRouter.complete`: ``text`` is ``None`` when the caller must fall back."""

    text: str | None
    latency_ms: float | None = None
    fallback_reason: str | None = None


def api_llm_label() -> str:
    """Human label for the API LLM, e.g. ``Groq gpt-oss-120b``."""
    settings = get_settings()
    name = settings.llm_model.rsplit("/", 1)[-1]
    return f"Groq {name}" if "groq.com" in settings.llm_base_url else name


class FinetunedRouter:
    """Fine-tuned endpoint client with a cached health check and a circuit breaker.

    Args:
        settings: Settings to read ``FINETUNED_*`` from. Defaults to :func:`get_settings`.
        client: OpenAI-compatible client (injected by tests). Built from settings if ``None``.
        http_client: ``httpx.Client`` for the health check (injected by tests).
        clock: Monotonic clock in seconds (injected by tests).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: Any | None = None,
        http_client: httpx.Client | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings: Settings = settings or get_settings()
        self._client = client
        self._http = http_client
        self._clock = clock
        self._lock = threading.Lock()
        self._health: FinetunedHealth | None = None
        self._health_at = float("-inf")
        self._open_until = float("-inf")

    @property
    def base_url(self) -> str:
        return self.settings.finetuned_base_url.rstrip("/")

    @property
    def circuit_open(self) -> bool:
        return self._clock() < self._open_until

    def _trip(self, reason: str) -> None:
        with self._lock:
            self._open_until = self._clock() + CIRCUIT_OPEN_S
        log.warning("Fine-tuned endpoint circuit opened for %.0f s: %s", CIRCUIT_OPEN_S, reason)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.settings.finetuned_api_key.get_secret_value()}"}

    def health(self) -> FinetunedHealth:
        """``GET {base}/models`` with a 3 s timeout; the result is cached for 60 s."""
        if not self.base_url:
            return FinetunedHealth(up=False, circuit_open=self.circuit_open)
        with self._lock:
            if self._health is not None and self._clock() - self._health_at < HEALTH_CACHE_S:
                return self._health.model_copy(update={"circuit_open": self.circuit_open})
        http = self._http or httpx.Client()
        start = time.perf_counter()
        try:
            resp = http.get(f"{self.base_url}/models", headers=self._headers(), timeout=HEALTH_TIMEOUT_S)
            up = resp.status_code == 200
        except httpx.HTTPError as exc:
            log.info("Fine-tuned health check failed: %s", exc)
            up = False
        finally:
            if self._http is None:
                http.close()
        latency = round((time.perf_counter() - start) * 1000, 1)
        result = FinetunedHealth(up=up, latency_ms=latency if up else None, circuit_open=self.circuit_open)
        with self._lock:
            self._health, self._health_at = result, self._clock()
        return result

    def _openai(self) -> Any:
        if self._client is None:
            self._client = openai.OpenAI(
                base_url=self.base_url,
                api_key=self.settings.finetuned_api_key.get_secret_value(),  # sent as the bearer token
                timeout=CALL_TIMEOUT_S,
                max_retries=0,
            )
        return self._client

    def complete(self, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> Completion:
        """Call the fine-tuned model, or explain why the caller should fall back.

        Never raises: every failure becomes ``Completion(text=None, fallback_reason=...)``.
        """
        if not self.base_url:
            return Completion(text=None, fallback_reason="FINETUNED_BASE_URL not set")
        if self.circuit_open:
            return Completion(text=None, fallback_reason="circuit open after recent failure")
        if not self.health().up:
            self._trip("health check failed")
            return Completion(text=None, fallback_reason="health check failed")
        start = time.perf_counter()
        try:
            with timed("llm", model="finetuned"):
                resp = self._openai().chat.completions.create(
                    model=self.settings.finetuned_model, messages=messages,
                    max_tokens=max_tokens, temperature=temperature, timeout=CALL_TIMEOUT_S,
                )
        except (openai.APITimeoutError, httpx.TimeoutException, TimeoutError) as exc:
            self._trip(f"timeout: {exc}")
            return Completion(text=None, fallback_reason="timeout")
        except Exception as exc:  # noqa: BLE001 — any endpoint failure falls back
            self._trip(repr(exc))
            return Completion(text=None, fallback_reason=f"error: {type(exc).__name__}")
        latency = round((time.perf_counter() - start) * 1000, 1)
        text = (resp.choices[0].message.content or "").strip() if resp.choices else ""
        if not text:
            return Completion(text=None, latency_ms=latency, fallback_reason="empty output")
        return Completion(text=text, latency_ms=latency)


_router: FinetunedRouter | None = None
_router_lock = threading.Lock()


def get_router() -> FinetunedRouter:
    """Process-wide router, so the health cache and circuit breaker are shared."""
    global _router
    with _router_lock:
        if _router is None:
            _router = FinetunedRouter()
        return _router


def build_pdf_prompt(
    question: str, rag_chunks: list[tuple[int, Chunk]], other_observations: list[str]
) -> list[dict[str, str]]:
    """System + user messages: question, top chunks as ``[Doc N, Sec S] text``, extra tool results."""
    seen: set[str] = set()
    top: list[tuple[int, Chunk]] = []
    for n, chunk in rag_chunks:
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            top.append((n, chunk))
        if len(top) == MAX_PROMPT_CHUNKS:
            break
    parts = [question.strip()]
    parts += [f"[Doc {n}, Sec {c.section or 'n/a'}] {sanitize_text(c.text).strip()}" for n, c in top]
    if other_observations:
        parts.append("Python results:\n" + "\n\n".join(o.strip() for o in other_observations))
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n\n".join(parts)}]


def generate_pdf_answer(
    question: str,
    rag_chunks: list[tuple[int, Chunk]],
    other_observations: list[str],
    groq_draft: str,
    router: FinetunedRouter | None = None,
) -> RoutedAnswer:
    """Answer a PDF-grounded question with the fine-tuned model, else return ``groq_draft``.

    Args:
        question: The user's message.
        rag_chunks: ``(Doc N, chunk)`` pairs retrieved this turn, best first; the top 4 are used.
        other_observations: python_interpreter outputs from the same turn.
        groq_draft: The API LLM's final answer, returned on any fallback.
        router: Defaults to :func:`get_router`.

    Returns:
        The answer with ``model_used``, ``latency_ms`` and ``fallback_reason``.
    """
    router = router or get_router()
    result = router.complete(
        build_pdf_prompt(question, rag_chunks, other_observations),
        max_tokens=ANSWER_MAX_TOKENS, temperature=ANSWER_TEMPERATURE,
    )
    if result.text is None:
        log.info("PDF answer falls back to API LLM: %s", result.fallback_reason)
        return RoutedAnswer(answer=groq_draft, model_used=api_llm_label(), fallback_reason=result.fallback_reason)
    return RoutedAnswer(answer=result.text, model_used=FINETUNED_LABEL, latency_ms=result.latency_ms)
