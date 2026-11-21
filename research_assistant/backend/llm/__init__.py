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

from backend.config import get_settings
from backend.llm.chain import AllModelsFailedError, ChatReply, FallbackChain, get_chain
from backend.runtime import timed


class LLM(Protocol):
    """Anything that turns chat messages into a reply."""

    def begin_turn(self) -> None:
        """Forget this thread's last reply (threadpool threads are reused across requests)."""
        self._local.reply = None
        self._local.finish_reason = None

    def generate(
        self, messages: list[dict[str, str]], max_tokens: int = 512, temperature: float = 0.0
    ) -> str: ...


class ApiLLM:
    """Agent chat model: every call goes through the ``AGENT_MODELS`` fallback chain.

    Args:
        chain: Fallback chain. Defaults to :func:`backend.llm.chain.get_chain`.

    Raises:
        EnvironmentError: If ``LLM_API_KEY`` is not set.
    """

    def __init__(self, chain: FallbackChain | None = None) -> None:
        settings = get_settings()
        if settings.llm_api_key is None or not settings.llm_api_key.get_secret_value():
            raise EnvironmentError("LLM_API_KEY is not set. Copy .env.example → .env and fill in your key.")
        self.chain: FallbackChain = chain or get_chain()
        # gpt-oss models are reasoning models: low effort leaves most of max_tokens for the reply.
        self.reasoning_effort: str = settings.llm_reasoning_effort
        self._local = threading.local()  # per-thread: requests run concurrently in FastAPI's threadpool

    @property
    def last_finish_reason(self) -> str | None:
        """``finish_reason`` of this thread's most recent :meth:`generate` call (e.g. ``stop``, ``length``)."""
        return getattr(self._local, "finish_reason", None)

    @property
    def last_reply(self) -> ChatReply | None:
        """This thread's most recent reply, including which chain model produced it."""
        return getattr(self._local, "reply", None)

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

        Raises:
            AllModelsFailedError: If every chain model failed.
        """
        extra: dict[str, object] = {"stop": stop} if stop else {}
        with timed("llm", model="agent-chain"):
            reply = self.chain.agent_chat(
                reasoning_effort=self.reasoning_effort,
                messages=messages, temperature=temperature, max_tokens=max_tokens, **extra,
            )
        self._local.finish_reason = reply.finish_reason
        self._local.reply = reply
        return reply.text


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


__all__ = ["LLM", "AllModelsFailedError", "ApiLLM", "get_llm"]
