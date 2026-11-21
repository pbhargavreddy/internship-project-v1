"""
test_llm.py
-----------
Offline test for the LLM provider switch in backend/llm/__init__.py.
"""

from __future__ import annotations

import pytest

import backend.llm as llm
from backend.config import Settings


def test_get_llm_returns_api_client_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(_env_file=None, llm_api_key="test-key")  # ignore the developer's .env
    monkeypatch.setattr(llm, "get_settings", lambda: settings)
    monkeypatch.setattr(llm, "_llm", None)

    client = llm.get_llm()

    assert settings.llm_provider == "api"
    assert isinstance(client, llm.ApiLLM)
    assert llm.get_llm() is client  # cached


def test_begin_turn_resets_last_reply_and_counts_usage_across_tool_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    import contextvars
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    from backend.llm.chain import FallbackChain, ModelSpec

    monkeypatch.setattr(llm, "get_settings", lambda: Settings(_env_file=None, llm_api_key="test-key"))
    reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi"), finish_reason="stop")],
                            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: reply)))
    spec = ModelSpec(provider="groq", model="m", base_url="https://x", api_key="k")
    api = llm.ApiLLM(chain=FallbackChain(specs=[spec], client_factory=lambda s: client))

    api.generate([{"role": "user", "content": "previous request"}])
    assert api.last_reply is not None
    api.begin_turn()
    assert api.last_reply is None  # a reused threadpool thread must not see the old model

    api.generate([])
    ctx = contextvars.copy_context()
    with ThreadPoolExecutor(2) as pool:  # like summarize_papers' section workers
        list(pool.map(lambda _: ctx.copy().run(api.generate, []), range(2)))
    assert api.turn_usage == {"calls": 3, "prompt_tokens": 30, "completion_tokens": 6}
