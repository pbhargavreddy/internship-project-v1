"""
chain.py
--------
Ordered fallback chain for the agent LLM.

``AGENT_MODELS`` lists ``provider:model`` entries (``groq:openai/gpt-oss-120b,...``).
``groq`` uses ``LLM_API_KEY`` and ``LLM_BASE_URL`` (or Groq's URL if ``LLM_BASE_URL`` is
not Groq). If ``FALLBACK_BASE_URL`` and ``FALLBACK_MODEL`` are set, that OpenAI-compatible
endpoint is appended as a last entry.

:func:`agent_chat` tries the models in order. It moves on after 429, 413, 5xx, timeouts,
connection errors and unknown or decommissioned model names. A 429 marks that (endpoint, account, model) as
exhausted for ``Retry-After`` seconds (60 s if the header is absent), so later turns skip
it immediately. Any other error, such as Groq's ``Tool choice is none`` 400, is raised
unchanged. When every model fails, :class:`AllModelsFailedError` is raised.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import openai
from pydantic import BaseModel, ConfigDict

from backend.config import Settings, get_settings

log = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_COOLDOWN_S = 60.0
CALL_TIMEOUT_S = 60.0
_MODEL_GONE = ("model_not_found", "does not exist", "decommissioned", "model_decommissioned")


class ModelSpec(BaseModel):
    """One entry of the chain."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    base_url: str
    api_key: str

    @property
    def key(self) -> str:
        """Label for logs and errors, e.g. ``groq:openai/gpt-oss-120b``. Never contains the API key."""
        return f"{self.provider}:{self.model}"

    @property
    def state_key(self) -> tuple[str, str, str]:
        """``(base_url, key fingerprint, model)``: the same model on another account has its own quota."""
        fingerprint = hashlib.sha256(self.api_key.encode()).hexdigest()[:12]
        return (self.base_url.rstrip("/"), fingerprint, self.model)

    @property
    def short_name(self) -> str:
        """``qwen3.8-27b`` for ``groq:qwen/qwen3.8-27b``."""
        return self.model.rsplit("/", 1)[-1]


class ChatReply(BaseModel):
    """Reply text plus the model that produced it."""

    text: str
    finish_reason: str | None
    model: ModelSpec
    fallback: bool  # True if a model earlier in the chain was skipped or failed


class AllModelsFailedError(RuntimeError):
    """Every model in the chain failed or is exhausted."""


def parse_chain(settings: Settings) -> list[ModelSpec]:
    """Build the chain from ``AGENT_MODELS`` plus the optional ``FALLBACK_*`` endpoint.

    Raises:
        ValueError: If an entry has an unknown provider.
    """
    key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else ""
    groq_url = settings.llm_base_url if "groq.com" in settings.llm_base_url else GROQ_BASE_URL
    specs: list[ModelSpec] = []
    for raw in (e.strip() for e in settings.agent_models.split(",")):
        if not raw:
            continue
        provider, sep, model = raw.partition(":")
        if not sep or provider != "groq":
            raise ValueError(f"AGENT_MODELS entry {raw!r}: expected 'groq:<model>'")
        specs.append(ModelSpec(provider="groq", model=model, base_url=groq_url, api_key=key))
    if settings.fallback_base_url and settings.fallback_model:
        fb_key = settings.fallback_api_key.get_secret_value() if settings.fallback_api_key else "none"
        specs.append(ModelSpec(provider="fallback", model=settings.fallback_model,
                               base_url=settings.fallback_base_url, api_key=fb_key))
    return specs


def _retry_after(exc: openai.APIStatusError) -> float:
    try:
        return max(float(exc.response.headers.get("retry-after", "")), 1.0)
    except (TypeError, ValueError):
        return DEFAULT_COOLDOWN_S


def _should_fall_through(exc: Exception) -> bool:
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError, TimeoutError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        if exc.status_code in (413, 429) or exc.status_code >= 500 or exc.status_code == 404:
            return True
        return exc.status_code == 400 and any(m in str(exc) for m in _MODEL_GONE)
    return False


class FallbackChain:
    """Tries each model in order; remembers rate-limited models across turns.

    Args:
        specs: Chain entries. Defaults to :func:`parse_chain` on the current settings.
        client_factory: Builds an OpenAI-compatible client for a spec (injected by tests).
        clock: Monotonic clock in seconds (injected by tests).
    """

    def __init__(
        self,
        specs: list[ModelSpec] | None = None,
        client_factory: Callable[[ModelSpec], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.specs: list[ModelSpec] = specs if specs is not None else parse_chain(get_settings())
        self._factory = client_factory or (lambda s: openai.OpenAI(
            base_url=s.base_url, api_key=s.api_key, timeout=CALL_TIMEOUT_S, max_retries=0))
        self._clock = clock
        self._clients: dict[tuple[str, str, str], Any] = {}
        self._exhausted_until: dict[tuple[str, str, str], float] = {}
        self._lock = threading.Lock()

    def _client(self, spec: ModelSpec) -> Any:
        with self._lock:
            if spec.state_key not in self._clients:
                self._clients[spec.state_key] = self._factory(spec)
            return self._clients[spec.state_key]

    def exhausted(self, spec: ModelSpec) -> bool:
        return self._clock() < self._exhausted_until.get(spec.state_key, float("-inf"))

    def agent_chat(self, reasoning_effort: str | None = None, **kwargs: Any) -> ChatReply:
        """``chat.completions.create(**kwargs)`` on the first model that works.

        Args:
            reasoning_effort: Sent only to gpt-oss models.
            **kwargs: ``messages``, ``max_tokens``, ``temperature``, ``stop``, ... (not ``model``).

        Returns:
            The reply, with the model that answered.

        Raises:
            AllModelsFailedError: If every model failed or is rate-limited.
            openai.APIError: Errors that a different model wouldn't fix (e.g. a bad request).
        """
        failures: list[str] = []
        for i, spec in enumerate(self.specs):
            if self.exhausted(spec):
                failures.append(f"{spec.key}: rate-limited")
                continue
            extra = {"reasoning_effort": reasoning_effort} if reasoning_effort and "gpt-oss" in spec.model else {}
            try:
                resp = self._client(spec).chat.completions.create(model=spec.model, **kwargs, **extra)
            except Exception as exc:  # noqa: BLE001 — classified below
                if not _should_fall_through(exc):
                    raise
                if isinstance(exc, openai.RateLimitError):
                    cooldown = _retry_after(exc)
                    with self._lock:
                        self._exhausted_until[spec.state_key] = self._clock() + cooldown
                    log.warning("Agent model %s rate-limited; skipping it for %.0f s", spec.key, cooldown)
                failures.append(f"{spec.key}: {type(exc).__name__}: {exc}")
                log.warning("Agent model %s failed (%s: %s); trying the next model",
                            spec.key, type(exc).__name__, str(exc)[:200])
                continue
            fallback = i > 0
            if failures:
                log.warning("Agent answer from fallback model %s after: %s", spec.key, "; ".join(failures)[:500])
            choice = resp.choices[0]
            return ChatReply(text=(choice.message.content or "").strip(), finish_reason=choice.finish_reason,
                             model=spec, fallback=fallback)
        raise AllModelsFailedError("All agent models failed: " + ("; ".join(failures) or "chain is empty"))


_chain: FallbackChain | None = None
_chain_lock = threading.Lock()


def get_chain() -> FallbackChain:
    """Process-wide chain, so rate-limit cooldowns are shared between requests."""
    global _chain
    with _chain_lock:
        if _chain is None:
            _chain = FallbackChain()
        return _chain


def agent_chat(**kwargs: Any) -> ChatReply:
    """:meth:`FallbackChain.agent_chat` on the process-wide chain."""
    return get_chain().agent_chat(**kwargs)
