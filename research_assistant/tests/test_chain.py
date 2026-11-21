"""
test_chain.py
-------------
Offline tests for the agent LLM fallback chain: fake clients raise or answer per model.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from backend.config import Settings
from backend.llm.chain import AllModelsFailedError, FallbackChain, ModelSpec, parse_chain

REQ = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def _err(status: int, msg: str = "boom", headers: dict[str, str] | None = None) -> openai.APIStatusError:
    resp = httpx.Response(status, request=REQ, headers=headers or {})
    cls = {429: openai.RateLimitError, 404: openai.NotFoundError, 400: openai.BadRequestError}.get(
        status, openai.InternalServerError if status >= 500 else openai.APIStatusError)
    return cls(msg, response=resp, body=None)


def _ok(text: str) -> Any:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")])


def _chain(behaviour: dict[str, list[Any]], now: list[float]) -> tuple[FallbackChain, list[str]]:
    """``behaviour[model]`` is a queue of exceptions / reply texts; ``calls`` records models tried."""
    calls: list[str] = []
    specs = [ModelSpec(provider="groq", model=m, base_url="https://x", api_key="k") for m in behaviour]

    def factory(spec: ModelSpec) -> Any:
        def create(model: str, **_kw: Any) -> Any:
            calls.append(model)
            item = behaviour[model].pop(0)
            if isinstance(item, Exception):
                raise item
            return _ok(item)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    return FallbackChain(specs=specs, client_factory=factory, clock=lambda: now[0]), calls


def test_parse_chain_default_and_optional_fallback() -> None:
    base = Settings(_env_file=None, llm_api_key="k")
    assert [s.key for s in parse_chain(base)] == [
        "groq:openai/gpt-oss-120b", "groq:qwen/qwen3.8-27b", "groq:openai/gpt-oss-20b"]
    extra = Settings(_env_file=None, llm_api_key="k", fallback_base_url="https://fb/v1", fallback_model="m")
    assert parse_chain(extra)[-1].key == "fallback:m"


def test_invalid_model_falls_through_to_next(caplog: pytest.LogCaptureFixture) -> None:
    chain, calls = _chain({"bad-model": [_err(404, "model_not_found")], "good": ["hello"]}, [0.0])
    reply = chain.agent_chat(messages=[])
    assert (reply.text, reply.model.model, reply.fallback) == ("hello", "good", True)
    assert "bad-model" in caplog.text


def test_429_marks_model_exhausted_for_retry_after() -> None:
    now = [0.0]
    chain, calls = _chain({"a": [_err(429, headers={"retry-after": "30"}), "a-ok"], "b": ["b1", "b2"]}, now)
    assert chain.agent_chat(messages=[]).model.model == "b"
    now[0] = 10.0
    assert chain.agent_chat(messages=[]).model.model == "b"  # a still cooling down: not called
    now[0] = 31.0
    assert chain.agent_chat(messages=[]).text == "a-ok"
    assert calls == ["a", "b", "b", "a"]


def test_all_fail_raises_and_bad_request_is_not_swallowed() -> None:
    chain, _ = _chain({"a": [_err(503)], "b": [openai.APITimeoutError(request=REQ)]}, [0.0])
    with pytest.raises(AllModelsFailedError):
        chain.agent_chat(messages=[])
    chain, calls = _chain({"a": [_err(400, "Tool choice is none")], "b": ["x"]}, [0.0])
    with pytest.raises(openai.BadRequestError):
        chain.agent_chat(messages=[])
    assert calls == ["a"]


def test_429_on_primary_uses_fallback_account_with_same_model(caplog: pytest.LogCaptureFixture) -> None:
    now = [0.0]
    calls: list[str] = []
    primary = ModelSpec(provider="groq", model="openai/gpt-oss-120b", base_url="https://api.groq.com/openai/v1", api_key="key-primary")
    friend = ModelSpec(provider="fallback", model="openai/gpt-oss-120b", base_url="https://api.groq.com/openai/v1", api_key="key-friend")

    def factory(spec: ModelSpec) -> Any:
        def create(model: str, **_kw: Any) -> Any:
            calls.append(spec.api_key)
            if spec.api_key == "key-primary":
                raise _err(429, headers={"retry-after": "30"})
            return _ok("from friend")
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    chain = FallbackChain(specs=[primary, friend], client_factory=factory, clock=lambda: now[0])
    assert primary.state_key != friend.state_key
    first = chain.agent_chat(messages=[])
    assert (first.text, first.model.provider, first.fallback) == ("from friend", "fallback", True)
    assert chain.exhausted(primary) and not chain.exhausted(friend)
    now[0] = 5.0
    chain.agent_chat(messages=[])
    assert calls == ["key-primary", "key-friend", "key-friend"]  # primary skipped while cooling down
    assert "key-primary" not in caplog.text and "key-friend" not in caplog.text


def _cooling_chain(retry_after: str, now: list[float], sleeps: list[float]) -> tuple[FallbackChain, list[str]]:
    calls: list[str] = []
    queue = {"a": [_err(429, headers={"retry-after": retry_after}), "a-ok"],
             "b": [_err(429, headers={"retry-after": retry_after}), "b-ok"]}
    specs = [ModelSpec(provider="groq", model=m, base_url="https://x", api_key="k") for m in queue]

    def factory(spec: ModelSpec) -> Any:
        def create(model: str, **_kw: Any) -> Any:
            calls.append(model)
            item = queue[model].pop(0)
            if isinstance(item, Exception):
                raise item
            return _ok(item)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    def sleep(s: float) -> None:
        sleeps.append(s)
        now[0] += s

    return FallbackChain(specs=specs, client_factory=factory, clock=lambda: now[0], sleep=sleep, max_wait_s=20), calls


def test_all_models_cooling_down_briefly_waits_once_and_retries() -> None:
    from backend.llm.chain import rate_limit_waits

    now, sleeps = [0.0], []
    chain, calls = _cooling_chain("8", now, sleeps)
    rate_limit_waits.set([])
    reply = chain.agent_chat(messages=[])
    assert reply.text == "a-ok" and calls == ["a", "b", "a"]
    assert sleeps == [8.5] and rate_limit_waits.get() == [8.5]


def test_long_retry_after_does_not_wait() -> None:
    now, sleeps = [0.0], []
    chain, _ = _cooling_chain("44", now, sleeps)
    with pytest.raises(AllModelsFailedError):
        chain.agent_chat(messages=[])
    assert sleeps == []


def test_request_too_large_does_not_wait() -> None:
    chain, _ = _chain({"a": [_err(429, headers={"retry-after": "5"})], "b": [_err(413, "Request too large")]}, [0.0])
    chain._sleep = lambda s: pytest.fail("must not wait when a model failed for a non-rate-limit reason")
    with pytest.raises(AllModelsFailedError):
        chain.agent_chat(messages=[])
