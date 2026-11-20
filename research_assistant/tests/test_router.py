"""
test_router.py
--------------
Offline tests for backend/llm/router.py and the agent's final-answer routing:
the Kaggle endpoint, its health check and the Groq LLM are all mocked.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

import backend.llm.router as router_mod
from backend.agent import ReActAgent, Session, SessionStore, Tool
from backend.agent.tools import RagSearchTool
from backend.config import Settings
from backend.ingestion import Chunk
from backend.llm.router import FINETUNED_LABEL, FinetunedRouter, generate_pdf_answer
from pydantic import BaseModel

CHUNK = Chunk(chunk_id="c1", doc_id="abc123", text="Transformers use self-attention.", page=3, section="Introduction")


class FakeKaggle:
    """Mimics ``OpenAI().chat.completions.create``; raises ``error`` if set."""

    def __init__(self, error: Exception | None = None, reply: str = "FT answer [Doc 1, Sec Introduction]") -> None:
        self.error, self.reply = error, reply
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw: Any) -> Any:
        self.calls.append(kw)
        if self.error:
            raise self.error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))])


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    s = Settings(_env_file=None, finetuned_base_url="http://kaggle.test/v1", finetuned_api_key="sekret",
                 llm_base_url="https://api.groq.com/openai/v1", llm_model="openai/gpt-oss-120b")
    monkeypatch.setattr(router_mod, "get_settings", lambda: s)
    return s


def _router(settings: Settings, kaggle: FakeKaggle, health_log: list[httpx.Request], clock: Clock | None = None):
    def health(req: httpx.Request) -> httpx.Response:
        health_log.append(req)
        return httpx.Response(200, json={"data": [{"id": "scholarpulse"}]})

    return FinetunedRouter(settings=settings, client=kaggle, clock=clock or Clock(),
                           http_client=httpx.Client(transport=httpx.MockTransport(health)))


def _groq(*replies: str) -> Any:
    it = iter(replies)
    return SimpleNamespace(generate=lambda messages, max_tokens=512, temperature=0.0: next(it))


def _action(tool: str, args: dict[str, Any]) -> str:
    return f"Thought: use {tool}\nAction: " + json.dumps({"tool": tool, "args": args})


def test_rag_turn_final_answer_comes_from_kaggle(settings: Settings, tmp_path) -> None:
    kaggle, health_log = FakeKaggle(), []
    agent = ReActAgent(
        llm=_groq(_action("rag_search", {"query": "attention"}), "Final Answer: Groq draft [Doc 1]"),
        store=SessionStore(tmp_path), router=_router(settings, kaggle, health_log),
        tool_factory=lambda s: [RagSearchTool(s, retrieve_fn=lambda q, k: [(CHUNK, 0.9)])],
    )

    result = agent.chat("s1", "How do transformers work?")

    assert result.answer == "FT answer [Doc 1, Sec Introduction]"
    assert result.model_used == FINETUNED_LABEL and result.latency_ms is not None and result.fallback_reason is None
    assert [(c.doc, c.page, c.section) for c in result.citations] == [(1, 3, "Introduction")]
    call = kaggle.calls[0]
    assert call["temperature"] == 0.2 and call["max_tokens"] == 1024 and call["timeout"] == 45
    assert "[Doc 1, Sec Introduction] Transformers use self-attention." in call["messages"][1]["content"]
    assert health_log[0].url.path == "/v1/models"
    assert health_log[0].headers["Authorization"] == "Bearer sekret"


class _StubPaperSearch(Tool):
    class Args(BaseModel):
        query: str

    name, description, args_model = "paper_search", "stub", Args

    def run(self, args: Any) -> str:
        return "1. Attention Is All You Need (2017)"


def test_paper_search_only_turn_stays_on_groq(settings: Settings, tmp_path) -> None:
    kaggle, health_log = FakeKaggle(), []
    agent = ReActAgent(
        llm=_groq(_action("paper_search", {"query": "transformers"}), "Final Answer: Groq answer"),
        store=SessionStore(tmp_path), router=_router(settings, kaggle, health_log),
        tool_factory=lambda s: [_StubPaperSearch()],
    )

    result = agent.chat("s1", "Find transformer papers")

    assert result.answer == "Groq answer" and result.model_used == "Groq gpt-oss-120b"
    assert result.fallback_reason is None
    assert kaggle.calls == [] and health_log == []


def test_timeout_falls_back_to_groq_draft(settings: Settings) -> None:
    timeout = openai.APITimeoutError(request=httpx.Request("POST", "http://kaggle.test/v1/chat/completions"))
    router = _router(settings, FakeKaggle(error=timeout), [])

    out = generate_pdf_answer("q", [(1, CHUNK)], [], "groq draft", router=router)

    assert out.answer == "groq draft" and out.model_used == "Groq gpt-oss-120b"
    assert out.fallback_reason == "timeout" and router.circuit_open


def test_open_circuit_skips_kaggle_for_120s(settings: Settings) -> None:
    clock, health_log = Clock(), []
    kaggle = FakeKaggle(error=RuntimeError("boom"))
    router = _router(settings, kaggle, health_log, clock)

    first = generate_pdf_answer("q", [(1, CHUNK)], [], "draft", router=router)
    clock.now += 100
    second = generate_pdf_answer("q", [(1, CHUNK)], [], "draft", router=router)

    assert first.fallback_reason == "error: RuntimeError"
    assert second.fallback_reason.startswith("circuit open") and second.answer == "draft"
    assert len(kaggle.calls) == 1 and len(health_log) == 1  # nothing sent while the circuit is open

    clock.now += 21  # 121 s after the failure
    kaggle.error = None
    third = generate_pdf_answer("q", [(1, CHUNK)], [], "draft", router=router)
    assert third.model_used == FINETUNED_LABEL and len(kaggle.calls) == 2


def test_paper_search_turn_routes_to_groq_even_with_rag_search(settings: Settings, tmp_path) -> None:
    kaggle, health_log = FakeKaggle(), []
    agent = ReActAgent(
        llm=_groq(_action("rag_search", {"query": "attention"}), _action("paper_search", {"query": "attention"}),
                  "Final Answer: Groq answer [Doc 1, Sec Introduction]"),
        store=SessionStore(tmp_path), router=_router(settings, kaggle, health_log),
        tool_factory=lambda s: [RagSearchTool(s, retrieve_fn=lambda q, k: [(CHUNK, 0.9)]), _StubPaperSearch()],
    )

    result = agent.chat("s1", "Compare my PDF with recent papers")

    assert result.answer == "Groq answer [Doc 1, Sec Introduction]"
    assert result.model_used == "Groq gpt-oss-120b" and result.fallback_reason is None
    assert kaggle.calls == [] and health_log == []
    assert [(c.doc, c.section) for c in result.citations] == [(1, "Introduction")]  # still from the rag chunks
