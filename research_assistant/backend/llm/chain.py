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

import contextvars
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
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    ratelimit: dict[str, str] = {}  # Groq x-ratelimit-* / retry-after response headers


RATELIMIT_HEADERS = (
    "x-ratelimit-limit-requests", "x-ratelimit-remaining-requests", "x-ratelimit-reset-requests",
    "x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens", "x-ratelimit-reset-tokens", "retry-after",
)


def _ratelimit_headers(headers: Any) -> dict[str, str]:
    if headers is None:
        return {}
    return {h.removeprefix("x-ratelimit-"): headers[h] for h in RATELIMIT_HEADERS if h in headers}


# Waits this turn spent on "every model is cooling down" (seconds), for the UI's agent steps.
# Reset per turn by ApiLLM.begin_turn; a ContextVar so tool worker threads share it.
rate_limit_waits: contextvars.ContextVar[list[float] | None] = contextvars.ContextVar("rate_limit_waits", default=None)


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
        sleep: Sleep function (injected by tests).
        max_wait_s: Longest "all models cooling down" wait before retrying once. Defaults to
            ``settings.rate_limit_max_wait_s``; 0 disables waiting.
    """

    def __init__(
        self,
        specs: list[ModelSpec] | None = None,
        client_factory: Callable[[ModelSpec], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        max_wait_s: float | None = None,
    ) -> None:
        self.specs: list[ModelSpec] = specs if specs is not None else parse_chain(get_settings())
        self._factory = client_factory or (lambda s: openai.OpenAI(
            base_url=s.base_url, api_key=s.api_key, timeout=CALL_TIMEOUT_S, max_retries=0))
        self._clock = clock
        self._sleep = sleep
        self.max_wait_s: float = max_wait_s if max_wait_s is not None else get_settings().rate_limit_max_wait_s
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

    def _cooldown_wait(self) -> float | None:
        """Seconds until the first model is usable, if EVERY model is cooling down after a 429."""
        now = self._clock()
        remaining = [self._exhausted_until.get(s.state_key, float("-inf")) - now for s in self.specs]
        if not remaining or any(r <= 0 for r in remaining):
            return None  # some model failed for another reason (413, 5xx, ...): waiting won't help
        return min(remaining)

    def agent_chat(self, reasoning_effort: str | None = None, **kwargs: Any) -> ChatReply:
        """:meth:`_try_chain`; if every model is cooling down for at most ``max_wait_s``,
        wait once and try the chain again instead of failing.

        Raises:
            AllModelsFailedError: If every model failed or is rate-limited (after the one wait).
            openai.APIError: Errors that a different model wouldn't fix (e.g. a bad request).
        """
        try:
            return self._try_chain(reasoning_effort, **kwargs)
        except AllModelsFailedError:
            wait = self._cooldown_wait()
            if wait is None or wait > self.max_wait_s:
                raise
        wait += 0.5  # Retry-After is rounded; don't arrive a moment early
        log.warning("Every agent model is rate-limited; waiting %.1f s, then retrying once", wait)
        waits = rate_limit_waits.get()
        if waits is not None:
            waits.append(round(wait, 1))
        self._sleep(wait)
        return self._try_chain(reasoning_effort, **kwargs)

    def _try_chain(self, reasoning_effort: str | None = None, **kwargs: Any) -> ChatReply:
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
            fingerprint = spec.state_key[1]
            completions = self._client(spec).chat.completions
            try:
                raw_api = getattr(completions, "with_raw_response", None)  # absent on test doubles
                if raw_api is not None:
                    raw = raw_api.create(model=spec.model, **kwargs, **extra)
                    resp, headers = raw.parse(), _ratelimit_headers(raw.headers)
                else:
                    resp, headers = completions.create(model=spec.model, **kwargs, **extra), {}
            except Exception as exc:  # noqa: BLE001 — classified below
                if not _should_fall_through(exc):
                    raise
                if isinstance(exc, openai.APIStatusError) and exc.status_code == 429:
                    # Groq's message names the limit: "... on tokens per minute (TPM): Limit .., Used .., Requested .."
                    log.warning("LLM 429 model=%s acct=%s headers=%s msg=%s", spec.key, fingerprint,
                                _ratelimit_headers(exc.response.headers), str(exc)[:400])
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
            usage = getattr(resp, "usage", None)
            prompt_toks = getattr(usage, "prompt_tokens", None)
            completion_toks = getattr(usage, "completion_tokens", None)
            log.info("LLM call model=%s acct=%s prompt_tokens=%s completion_tokens=%s ratelimit=%s",
                     spec.key, fingerprint, prompt_toks, completion_toks, headers)
            return ChatReply(text=(choice.message.content or "").strip(), finish_reason=choice.finish_reason,
                             model=spec, fallback=fallback, prompt_tokens=prompt_toks,
                             completion_tokens=completion_toks, ratelimit=headers)
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
