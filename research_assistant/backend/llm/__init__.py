"""
llm
---
Chat-model backend behind one interface: :class:`ApiLLM`, any OpenAI-compatible
endpoint (``LLM_PROVIDER=api``, the only allowed value).

The backend never imports ``backend.llm.local_model`` (transformers Qwen on CPU);
that module is for the notebooks only. ``LLM_PROVIDER=local`` fails settings validation.
"""

from __future__ import annotations

import threading
from typing import Protocol

from openai import OpenAI

from backend.config import get_settings
from backend.runtime import timed


class LLM(Protocol):
    """Anything that turns chat messages into a reply."""

    def generate(
        self, messages: list[dict[str, str]], max_tokens: int = 512, temperature: float = 0.0
    ) -> str: ...


class ApiLLM:
    """OpenAI-compatible chat endpoint (OpenAI, Gemini compat, Groq, Ollama).

    Raises:
        EnvironmentError: If ``LLM_API_KEY`` is not set.
    """

    def __init__(self) -> None:
        settings = get_settings()
        if settings.llm_api_key is None or not settings.llm_api_key.get_secret_value():
            raise EnvironmentError("LLM_API_KEY is not set. Copy .env.example → .env and fill in your key.")
        self.model: str = settings.llm_model
        # gpt-oss models are reasoning models: low effort leaves most of max_tokens for the reply.
        self.reasoning_effort: str | None = settings.llm_reasoning_effort if "gpt-oss" in self.model.lower() else None
        self.client = OpenAI(api_key=settings.llm_api_key.get_secret_value(), base_url=settings.llm_base_url)
        self._local = threading.local()  # per-thread: requests run concurrently in FastAPI's threadpool

    @property
    def last_finish_reason(self) -> str | None:
        """``finish_reason`` of this thread's most recent :meth:`generate` call (e.g. ``stop``, ``length``)."""
        return getattr(self._local, "finish_reason", None)

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        """One chat completion.

        Args:
            messages: Chat messages.
            max_tokens: Completion token limit.
            temperature: Sampling temperature.
            stop: Optional stop sequences (the agent stops at ``"Observation:"``).

        Returns:
            The reply text, stripped.
        """
        extra: dict[str, object] = {}
        if stop:
            extra["stop"] = stop
        if self.reasoning_effort:
            extra["reasoning_effort"] = self.reasoning_effort
        with timed("llm", model=self.model):
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens, **extra
            )
        choice = response.choices[0]
        self._local.finish_reason = choice.finish_reason
        return (choice.message.content or "").strip()


_llm: LLM | None = None


def get_llm() -> LLM:
    """Return the process-wide API LLM (built on first use).

    Raises:
        EnvironmentError: If ``LLM_API_KEY`` is not set.
    """
    global _llm
    if _llm is None:
        _llm = ApiLLM()
    return _llm


__all__ = ["LLM", "ApiLLM", "get_llm"]
