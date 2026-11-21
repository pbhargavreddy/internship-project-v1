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

import httpx
import openai
import pytest
from pydantic import BaseModel

from backend.agent import ReActAgent, Session, SessionStore, Tool, run_python
from backend.agent.react import AGENT_UNAVAILABLE, FINAL_RETRY_PROMPT
from backend.agent.tools import PaperRef, PaperSearchArgs, PythonInterpreterTool, RagSearchTool
from backend.llm import AllModelsFailedError
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


def test_no_tool_turn_still_returns_agent_log_notes(tmp_path: Path) -> None:
    llm = FakeLLM(_replies("", "Thought: just a greeting\nFinal Answer: Hello!"))
    llm.last_finish_reason = "stop"

    result = _agent(llm, tmp_path, raw_log_path=tmp_path / "raw.log").chat("s1", "Hello agent")

    assert result.steps == []
    assert [(n.step, n.kind) for n in result.notes] == [(1, "reask"), (1, "final_thought"), (1, "routing")]
    assert "finish_reason=stop" in result.notes[0].text and "<empty>" in result.notes[0].text
    assert result.notes[1].text == "just a greeting"
    assert result.notes[2].text.startswith("Final answer by ")


# --- Groq "Tool choice is none" 400 handling ---------------------------------

def _tool_call_400() -> openai.BadRequestError:
    resp = httpx.Response(400, request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"))
    return openai.BadRequestError("Error code: 400 - Tool choice is none, but model called a tool", response=resp, body=None)


def test_tool_call_400_retries_in_plain_text(tmp_path: Path) -> None:
    def script(messages: list[dict[str, str]]) -> str:
        if messages[-1]["content"] == FINAL_RETRY_PROMPT:
            return "plain text answer"
        raise _tool_call_400()

    result = _agent(FakeLLM(script), tmp_path).chat("g1", "find papers on attention")
    assert result.answer == "plain text answer"


def test_tool_call_400_twice_falls_back_to_papers(tmp_path: Path) -> None:
    ref = PaperRef(id="arxiv:1706.03762", title="Attention Is All You Need", year=2017,
                   authors=["Vaswani"], url="https://arxiv.org/abs/1706.03762", source="arxiv")

    class FakePaperTool(Tool):
        name = "paper_search"
        description = "fake"
        args_model = BaseModel

        def __init__(self) -> None:
            self.papers = [ref]

        def run(self, args: BaseModel) -> str:
            return "1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)"

    calls = iter([f'Thought: search\nAction: {json.dumps({"tool": "paper_search", "args": {}})}'])

    def script(_messages: list[dict[str, str]]) -> str:
        reply = next(calls, None)
        if reply is None:
            raise _tool_call_400()
        return reply

    agent = ReActAgent(llm=FakeLLM(script), store=SessionStore(tmp_path), tool_factory=lambda s: [FakePaperTool()],
                       router=FinetunedRouter(settings=Settings(_env_file=None)))
    result = agent.chat("g2", "find papers on attention")
    assert "[Attention Is All You Need](https://arxiv.org/abs/1706.03762)" in result.answer


def test_other_bad_request_still_raises(tmp_path: Path) -> None:
    resp = httpx.Response(400, request=httpx.Request("POST", "https://x"))

    def script(_messages: list[dict[str, str]]) -> str:
        raise openai.BadRequestError("context length exceeded", response=resp, body=None)

    with pytest.raises(openai.BadRequestError):
        _agent(FakeLLM(script), tmp_path).chat("g3", "hi")


# --- degraded mode (every agent model failed) ---------------------------------

class _DownLLM:
    def generate(self, messages: list[dict[str, str]], max_tokens: int = 512, temperature: float = 0.0) -> str:
        raise AllModelsFailedError("All agent models failed: groq:a: RateLimitError")


def test_degraded_paper_search_and_plain_message(tmp_path: Path) -> None:
    ref = PaperRef(id="arxiv:1", title="Attention", url="https://arxiv.org/abs/1", source="arxiv")
    queries: list[str] = []

    class Papers(Tool):
        name = "paper_search"
        description = "fake"
        args_model = PaperSearchArgs

        def __init__(self) -> None:
            self.papers: list[PaperRef] = []

        def run(self, args: PaperSearchArgs) -> str:
            queries.append(args.query)
            self.papers.append(ref)
            return "1. [Attention](https://arxiv.org/abs/1)"

    agent = ReActAgent(llm=_DownLLM(), store=SessionStore(tmp_path), tool_factory=lambda s: [Papers()],
                       router=FinetunedRouter(settings=Settings(_env_file=None)))
    result = agent.chat("d1", "find papers on attention")
    assert result.degraded and result.papers == [ref] and queries == ["attention"]

    result = agent.chat("d1", "what is 2+2?")
    assert result.degraded and result.answer == AGENT_UNAVAILABLE


def test_degraded_pdf_question_without_finetuned_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import backend.agent.react as react
    monkeypatch.setattr(react, "load_doc_names", lambda chat_id=None: {"abc123": "R6.pdf"})
    agent = _agent(_DownLLM(), tmp_path)  # offline router: FINETUNED_BASE_URL not set
    result = agent.chat("d2", "Summarize R6.pdf")
    assert result.degraded and result.steps[0].tool == "rag_search"
    assert "fine-tuned PDF model is unreachable" in result.answer and "Groq" not in result.answer


# --- per-chat conversation memory ---------------------------------------------

def test_history_is_trimmed_oldest_first_to_the_token_cap() -> None:
    from backend.agent.memory import Turn, trim_history

    turns = [Turn(user=f"q{i}", assistant=("word " * 400) + f"a{i}") for i in range(6)]  # ~400 tokens each
    kept = trim_history(turns, max_messages=10, max_tokens=1000)
    assert [t.user for t in kept] == ["q4", "q5"]  # 5 turns allowed by count, 2 fit the cap
    assert [t.user for t in trim_history(turns, max_messages=4, max_tokens=10_000)] == ["q4", "q5"]
    huge = trim_history([Turn(user="q", assistant="word " * 5000)], max_messages=10, max_tokens=100)
    assert len(huge) == 1 and len(huge[0].assistant) < 1000  # newest turn kept, answer cut


def test_follow_up_is_rewritten_for_retrieval_and_finetuned_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import backend.agent.react as react
    from backend.agent.memory import Turn

    store = SessionStore(tmp_path)
    session = store.load("m1")
    session.turns.append(Turn(user="summarize R6.pdf", assistant="R6.pdf proposes a dehazing transformer."))
    store.save(session)

    queries: list[str] = []

    def retrieve(query: str, k: int) -> list[tuple[Chunk, float]]:
        queries.append(query)
        return _fake_retrieve(query, k)

    seen: dict[str, str] = {}

    def fake_pdf_answer(question, rag_chunks, other, draft, router=None):  # noqa: ANN001
        seen["question"] = question
        return react.RoutedAnswer(answer=draft, model_used="test")

    monkeypatch.setattr(react, "generate_pdf_answer", fake_pdf_answer)
    prompts: list[list[dict[str, str]]] = []
    replies = iter([
        f'Thought: look\nAction: {json.dumps({"tool": "rag_search", "args": {"query": "dataset it used"}})}',
        "Which dataset did R6.pdf use?",  # the rewrite call
        "Final Answer: It used RESIDE [Doc 1, Sec Introduction].",
    ])

    def script(messages: list[dict[str, str]]) -> str:
        prompts.append(messages)
        return next(replies)

    agent = ReActAgent(llm=FakeLLM(script), store=store, router=FinetunedRouter(settings=Settings(_env_file=None)),
                       tool_factory=lambda s: [RagSearchTool(s, retrieve_fn=retrieve)])
    result = agent.chat("m1", "what dataset did it use?")

    assert queries == ["Which dataset did R6.pdf use?"]
    assert seen["question"] == "Which dataset did R6.pdf use?"
    assert prompts[0][1:3] == [{"role": "user", "content": "summarize R6.pdf"},
                               {"role": "assistant", "content": "R6.pdf proposes a dehazing transformer."}]
    assert any(n.kind == "standalone" for n in result.notes)


def test_no_rewrite_call_without_history(tmp_path: Path) -> None:
    action = json.dumps({"tool": "rag_search", "args": {"query": "attention"}})
    llm = FakeLLM(_replies(f"Thought: s\nAction: {action}", "Final Answer: done"))
    _agent(llm, tmp_path).chat("m2", "How does attention work?")
    assert llm.calls == 2


def test_doc_filtered_query_drops_the_filename() -> None:
    from backend.agent.tools import GENERIC_DOC_QUERY, strip_doc_name

    assert strip_doc_name("What dataset did the paper R6.pdf use?", "R6.pdf") == "What dataset did the paper use?"
    assert strip_doc_name("what does R6 say about FPGA latency", "R6.pdf") == "what does say about FPGA latency"
    assert strip_doc_name("R6.pdf", "R6.pdf") == GENERIC_DOC_QUERY
    assert strip_doc_name("dataset of doc 3", "Doc 3") == "dataset of doc 3"


def test_request_over_budget_trims_history_first_then_old_tool_results() -> None:
    from backend.agent.memory import count_tokens
    from backend.agent.react import CURRENT_PREFIX, OBS_PREFIX, fit_to_budget

    big = "word " * 1000  # ~1000 tokens
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "old q"}, {"role": "assistant", "content": big},
            {"role": "user", "content": "q2"}, {"role": "assistant", "content": big},
            {"role": "user", "content": CURRENT_PREFIX + "find papers"},
            {"role": "assistant", "content": "Thought: search\nAction: {}"},
            {"role": "user", "content": OBS_PREFIX + big},
            {"role": "assistant", "content": "Thought: again\nAction: {}"},
            {"role": "user", "content": OBS_PREFIX + "latest " * 300}]
    total = lambda ms: sum(count_tokens(m["content"]) for m in ms)  # noqa: E731

    one_pair = fit_to_budget(msgs, total(msgs) - 500)
    assert [m["content"] for m in one_pair[1:3]] == ["q2", big]  # only the oldest history pair dropped

    tight = fit_to_budget(msgs, 900)
    assert tight[1]["content"].startswith(CURRENT_PREFIX)  # all history gone
    assert len(tight[3]["content"]) < 700 and tight[-1]["content"].startswith(OBS_PREFIX + "latest")
    assert msgs[2]["content"] == big  # input not modified
