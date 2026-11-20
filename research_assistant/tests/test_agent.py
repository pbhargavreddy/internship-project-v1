"""
test_agent.py
-------------
Offline tests for the ReAct agent: a scripted fake LLM drives the loop, and
rag_search gets a stub retriever, so no keys, network or vector store are needed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from backend.agent import ReActAgent, Session, SessionStore, Tool, run_python
from backend.agent.tools import PythonInterpreterTool, RagSearchTool
from backend.config import Settings
from backend.ingestion import Chunk
from backend.llm.router import FinetunedRouter


class FakeLLM:
    """Returns scripted replies; ``script`` gets the message list and returns the next reply."""

    def __init__(self, script: Callable[[list[dict[str, str]]], str]) -> None:
        self.script = script
        self.calls = 0

    def generate(self, messages: list[dict[str, str]], max_tokens: int = 512, temperature: float = 0.0) -> str:
        self.calls += 1
        return self.script(messages)


def _replies(*replies: str) -> Callable[[list[dict[str, str]]], str]:
    it = iter(replies)
    return lambda _messages: next(it)


def _fake_retrieve(query: str, k: int) -> list[tuple[Chunk, float]]:
    chunk = Chunk(chunk_id="c1", doc_id="abc123", text="Transformers use self-attention.", page=3, section="Introduction")
    return [(chunk, 0.9)]


def _agent(llm: FakeLLM, tmp_path: Path, **kw: object) -> ReActAgent:
    def tools(session: Session) -> list[Tool]:
        return [RagSearchTool(session, retrieve_fn=_fake_retrieve), PythonInterpreterTool()]

    offline = FinetunedRouter(settings=Settings(_env_file=None))  # no FINETUNED_BASE_URL: Groq answers
    return ReActAgent(llm=llm, store=SessionStore(tmp_path), tool_factory=tools, router=offline, **kw)


def test_tool_call_then_final_answer(tmp_path: Path) -> None:
    action = json.dumps({"tool": "rag_search", "args": {"query": "attention"}})
    llm = FakeLLM(_replies(
        f"Thought: search local docs\nAction: {action}",
        "Thought: done\nFinal Answer: Transformers rely on self-attention [Doc 1, Sec Introduction].",
    ))
    result = _agent(llm, tmp_path).chat("s1", "How do transformers work?")

    assert result.answer.startswith("Transformers rely")
    assert len(result.steps) == 1 and result.steps[0].tool == "rag_search"
    assert result.steps[0].observation.startswith("[Doc 1, Sec Introduction, p.3]")
    assert [(c.doc, c.page, c.section) for c in result.citations] == [(1, 3, "Introduction")]
    saved = SessionStore(tmp_path).load("s1")
    assert len(saved.turns) == 1 and saved.doc_numbers == {"abc123": 1}


def test_bad_json_is_fed_back_and_recovers(tmp_path: Path) -> None:
    llm = FakeLLM(_replies(
        'Thought: oops\nAction: {"tool": "rag_search", "args": {',
        'Thought: unknown\nAction: {"tool": "nope", "args": {}}',
        "Final Answer: recovered",
    ))
    result = _agent(llm, tmp_path).chat("s2", "hi")

    assert result.answer == "recovered"
    assert "Invalid Action JSON" in result.steps[0].observation
    assert "unknown tool 'nope'" in result.steps[1].observation


def test_step_limit_forces_final_answer(tmp_path: Path) -> None:
    action = json.dumps({"tool": "rag_search", "args": {"query": "x"}})

    def script(messages: list[dict[str, str]]) -> str:
        if "step limit" in messages[-1]["content"]:
            return "Final Answer: forced"
        return f"Thought: again\nAction: {action}"

    llm = FakeLLM(script)
    result = _agent(llm, tmp_path, max_steps=6).chat("s3", "loop forever")

    assert result.answer == "forced"
    assert len(result.steps) == 6 and llm.calls == 7


def test_python_interpreter_timeout(tmp_path: Path) -> None:
    out = run_python("print('start', flush=True)\nwhile True:\n    pass", timeout=1, output_dir=tmp_path)
    assert "timed out after 1 s" in out and "start" in out


def test_citations_use_registered_filename_not_doc_id(tmp_path: Path, monkeypatch) -> None:
    import backend.ingestion.names as names
    from backend.agent import Citation
    from backend.agent.react import extract_citations

    registry = tmp_path / "names.json"
    monkeypatch.setattr(names, "get_settings", lambda: Settings(_env_file=None, doc_names_file=registry))
    names.save_doc_names({"abc123": "R6.pdf"})  # e.g. recorded by an upload or another session's ingest

    action = json.dumps({"tool": "rag_search", "args": {"query": "attention"}})
    llm = FakeLLM(_replies(f"Thought: search\nAction: {action}", "Final Answer: Self-attention [Doc 1, Sec Introduction]."))
    result = _agent(llm, tmp_path / "sessions").chat("s1", "How do transformers work?")
    assert [(c.title, c.page, c.section) for c in result.citations] == [("R6.pdf", 3, "Introduction")]

    # Sources saved before the fix carry a placeholder title; it is replaced on lookup.
    stale = Session(session_id="s2", doc_numbers={"abc123": 1}, sources=[Citation(doc=1, title="Document abc123")])
    assert extract_citations("See [Doc 1].", stale)[0].title == "R6.pdf"


def test_parser_accepts_action_on_same_line_as_thought() -> None:
    from backend.agent.react import parse_reply

    reply = 'Thought: I should search. Action: {"tool": "rag_search", "args": {"query": "a {b}"}} done'
    parsed = parse_reply(reply)

    assert parsed.error is None and parsed.thought == "I should search."
    assert parsed.action.tool == "rag_search" and parsed.action.args == {"query": "a {b}"}


def test_rag_search_k_is_clamped_to_ten(tmp_path: Path) -> None:
    seen: list[int] = []

    def retrieve(query: str, k: int) -> list[tuple[Chunk, float]]:
        seen.append(k)
        return _fake_retrieve(query, k)

    tool = RagSearchTool(Session(session_id="s"), retrieve_fn=retrieve)
    tool({"query": "attention", "k": 20})
    tool({"query": "attention", "k": 0})

    assert seen == [10, 1]


def test_unparseable_reply_is_logged_with_finish_reason(tmp_path: Path) -> None:
    llm = FakeLLM(_replies("I think the answer is", "Thought: ok\nFinal Answer: done"))
    llm.last_finish_reason = "length"
    log_file = tmp_path / "logs" / "agent_raw.log"

    result = _agent(llm, tmp_path, raw_log_path=log_file).chat("s1", "question")

    text = log_file.read_text(encoding="utf-8")
    assert result.answer == "done"
    assert "session=s1 step=1 finish_reason=length" in text
    assert "I think the answer is" in text and text.count("=== ") == 1  # the good reply isn't logged


def test_thought_without_action_is_reasked_once_without_a_step(tmp_path: Path) -> None:
    seen: list[list[dict[str, str]]] = []
    replies = iter(["Thought: I need to look this up.", "Final Answer: done"])

    def script(messages: list[dict[str, str]]) -> str:
        seen.append(list(messages))
        return next(replies)

    llm = FakeLLM(script)
    result = _agent(llm, tmp_path, raw_log_path=tmp_path / "raw.log").chat("s1", "question")

    assert result.answer == "done" and result.steps == [] and llm.calls == 2
    assert seen[1][-1] == {"role": "user", "content": "Continue: give the Action JSON or Final Answer."}
    assert seen[0][-1]["content"].startswith("CURRENT user message:")


def test_paper_search_results_are_returned_as_papers_list(tmp_path: Path, monkeypatch) -> None:
    import backend.tools.web_search as web_search
    from backend.agent.tools import PaperSearchTool

    paper = web_search.Paper(id="arxiv:1706.03762", title="Attention Is All You Need", authors=["A. Vaswani"],
                             year=2017, url="https://arxiv.org/abs/1706.03762",
                             pdf_url="https://arxiv.org/pdf/1706.03762", source="arxiv", citation_count=100000)
    monkeypatch.setattr(web_search, "search_papers", lambda q, max_results: [paper])
    action = json.dumps({"tool": "paper_search", "args": {"query": "transformers"}})
    observations: list[str] = []

    def script(messages: list[dict[str, str]]) -> str:
        if len(observations) == 0 and messages[-1]["content"].startswith("Observation:"):
            observations.append(messages[-1]["content"])
            return "Final Answer: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017)"
        return f"Thought: search\nAction: {action}"

    agent = ReActAgent(llm=FakeLLM(script), store=SessionStore(tmp_path), tool_factory=lambda s: [PaperSearchTool(s)],
                       router=FinetunedRouter(settings=Settings(_env_file=None)), raw_log_path=tmp_path / "raw.log")
    result = agent.chat("s1", "Find the transformer paper")

    assert [p.model_dump() for p in result.papers] == [{
        "id": "arxiv:1706.03762", "title": "Attention Is All You Need", "year": 2017, "authors": ["A. Vaswani"],
        "url": "https://arxiv.org/abs/1706.03762", "pdf_url": "https://arxiv.org/pdf/1706.03762",
        "source": "arxiv", "citation_count": 100000,
    }]
    assert "[Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017)" in observations[0]
    assert "PDF: https://arxiv.org/pdf/1706.03762" in observations[0]
