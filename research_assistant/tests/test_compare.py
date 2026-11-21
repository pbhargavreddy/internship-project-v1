"""
test_compare.py
---------------
Offline tests for compare_papers, resolve_docs and the clarification follow-up:
a scripted fake LLM, a stub per-document retriever and a temp names registry.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

import backend.ingestion.names as names
from backend.agent import ReActAgent, Session, SessionStore, Tool, Turn
from backend.agent.compare import COMPARE_ERROR, SYSTEM as COMPARE_SYSTEM, ComparePapersTool
from backend.agent.doc_resolver import resolve_docs
from backend.agent.react import REWRITE_PROMPT
from backend.agent.tools import RagSearchTool
from backend.config import Settings
from backend.ingestion import Chunk
from backend.llm.router import FinetunedRouter

CHAT = "c1"


@pytest.fixture
def docs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Registers filenames for chat ``c1`` in a temp registry (doc_id = name without .pdf)."""
    registry = tmp_path / "names.json"
    monkeypatch.setattr(names, "get_settings", lambda: Settings(_env_file=None, doc_names_file=registry))

    def add(*filenames: str) -> None:
        for f in filenames:
            names.attach_doc(f.removesuffix(".pdf"), f, CHAT)

    return add


def _retrieve(query: str, k: int, doc_ids: list[str] | None) -> list[tuple[Chunk, float]]:
    d = doc_ids[0]
    return [(Chunk(chunk_id=f"{d}-{i}", doc_id=d, text=f"{d} passage about {query}. " * 30, page=1, section="Method"), 1.0)
            for i in range(k)]


def _table(*docs: str, extra: int = 0) -> str:
    rows = [{"aspect": a, **{d: f"{a} of {d} [Doc {i}, Sec Method]" for i, d in enumerate(docs, 1)}}
            for a in ["Objective", "Method", "Dataset", "Results / metrics", "Limitations"]]
    return json.dumps({"rows": rows, "summaries": {d: f"{d} summary." for d in docs},
                       "differences": ["They differ."] + [f"Difference {i}: " + "x" * 300 for i in range(extra)]})


class FakeLLM:
    """Routes by system prompt: rewrite, compare, or the ReAct loop."""

    def __init__(self, rewrite: str = "", compare: list[str] | None = None) -> None:
        self.rewrite, self.compare = rewrite, list(compare or [])
        self.calls: list[str] = []

    def generate(self, messages: list[dict[str, str]], max_tokens: int = 512, temperature: float = 0.0) -> str:
        system = messages[0]["content"]
        if system == REWRITE_PROMPT:
            self.calls.append("rewrite")
            return self.rewrite
        if system == COMPARE_SYSTEM:
            self.calls.append("compare")
            return self.compare.pop(0)
        self.calls.append("react")
        return "Final Answer: not a comparison"


def _agent(llm: FakeLLM, store: SessionStore) -> ReActAgent:
    def tools(session: Session) -> list[Tool]:
        return [RagSearchTool(session, retrieve_fn=lambda q, k: []),
                ComparePapersTool(session, doc_retrieve_fn=_retrieve, llm=llm)]

    return ReActAgent(llm=llm, store=store, tool_factory=tools, router=FinetunedRouter(settings=Settings(_env_file=None)))


def test_answer_to_clarifying_question_keeps_the_compare_task(tmp_path: Path, docs) -> None:
    docs("journal1.pdf", "journal2.pdf", "journal3.pdf")
    store = SessionStore(tmp_path / "sessions")
    session = store.load(CHAT)
    session.turns.append(Turn(user="compare these 2 paper and summarize them", assistant="Which two papers?"))
    store.save(session)
    llm = FakeLLM(rewrite="Compare journal1.pdf and journal2.pdf and summarize them.",
                  compare=[_table("journal1.pdf", "journal2.pdf")])

    result = _agent(llm, store).chat(CHAT, "journal1 and journal2")

    assert result.steps[0].tool == "compare_papers"
    assert result.steps[0].args == {"docs": ["journal1.pdf", "journal2.pdf"]}
    assert "| Aspect | journal1.pdf | journal2.pdf |" in result.answer and "### Key differences" in result.answer
    assert llm.calls == ["rewrite", "compare"]  # no ReAct routing, no final-answer call


def test_two_docs_need_no_clarification(tmp_path: Path, docs) -> None:
    docs("journal1.pdf", "journal2.pdf")
    llm = FakeLLM(compare=[_table("journal1.pdf", "journal2.pdf")])
    result = _agent(llm, SessionStore(tmp_path / "sessions")).chat(CHAT, "compare these papers")
    assert "| Aspect | journal1.pdf | journal2.pdf |" in result.answer
    assert llm.calls == ["compare"] and result.citations


def test_three_docs_ask_which_and_resume_from_pending(tmp_path: Path, docs) -> None:
    from backend.storage import chats

    docs("journal1.pdf", "journal2.pdf", "journal3.pdf")
    chats.set_chat_store(chats.LocalChatStore(tmp_path / "chats"))
    try:
        chat_id = chats.create_chat("t").chat_id
        for f in ("journal1.pdf", "journal2.pdf", "journal3.pdf"):
            names.attach_doc(f.removesuffix(".pdf"), f, chat_id)
        store = chats.ChatHistorySessionStore([], root=tmp_path / "sessions")
        llm = FakeLLM(compare=[_table("journal1.pdf", "journal3.pdf")])

        ask = _agent(llm, store).chat(chat_id, "compare these papers")
        assert all(f in ask.answer for f in ("journal1.pdf", "journal2.pdf", "journal3.pdf"))
        assert chats.get_chat(chat_id).pending_action.model_dump() == {"tool": "compare_papers", "missing": "docs"}

        done = _agent(llm, store).chat(chat_id, "Journal 1 and doc 3")
        assert "| Aspect | journal1.pdf | journal3.pdf |" in done.answer
        assert chats.get_chat(chat_id).pending_action is None
        assert llm.calls == ["compare"]
    finally:
        chats.set_chat_store(None)


def test_resolve_docs_spellings(docs) -> None:
    docs("journal1.pdf", "Journal_10.pdf", "deep-net.pdf")
    assert resolve_docs("journal1", CHAT) == ["journal1.pdf"]
    assert resolve_docs("Journal 1 vs JOURNAL1.PDF", CHAT) == ["journal1.pdf"]
    assert resolve_docs("journal 10 and Deep Net", CHAT) == ["Journal_10.pdf", "deep-net.pdf"]
    assert resolve_docs("paper 3 and doc 1", CHAT) == ["deep-net.pdf", "journal1.pdf"]
    assert resolve_docs("journal2", CHAT) == []


def test_broken_json_is_retried_once_then_clean_error(tmp_path: Path, docs) -> None:
    docs("journal1.pdf", "journal2.pdf")
    llm = FakeLLM(compare=['{"rows": [', "sorry, no json"])
    result = _agent(llm, SessionStore(tmp_path / "sessions")).chat(CHAT, "compare these papers")
    assert result.answer == COMPARE_ERROR and llm.calls == ["compare", "compare"]


def test_long_table_is_not_cut_at_4000_chars(tmp_path: Path, docs) -> None:
    docs("journal1.pdf", "journal2.pdf")
    llm = FakeLLM(compare=[_table("journal1.pdf", "journal2.pdf", extra=20)])
    result = _agent(llm, SessionStore(tmp_path / "sessions")).chat(CHAT, "journal1 vs journal2")
    assert len(result.answer) > 4000 and "truncated" not in result.answer
    assert result.answer.rstrip().endswith("x" * 300)
