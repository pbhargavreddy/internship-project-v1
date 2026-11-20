"""
test_summarize_tool.py
----------------------
Offline tests for the agent's section-by-section summarize_papers tool: the
fine-tuned endpoint (behind the router) and the API-LLM fallback are mocked.
"""

from __future__ import annotations

import re
import threading
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import backend.llm.router as router_mod
from backend.agent import Session
from backend.agent.tools import SummarizePapersTool, group_sections
from backend.config import Settings
from backend.ingestion import Chunk
from backend.llm.router import FinetunedRouter

SECTIONS = ["Abstract", "3 Proposed Method", "4.1 Datasets", "5 Experimental Results", "6 Conclusion"]
CHUNKS = [
    Chunk(chunk_id=f"c{i}", doc_id="d6", text=f"text of {sec}", page=i + 1, section=sec)
    for i, sec in enumerate(SECTIONS)
]
_PART_RE = re.compile(r"Summarize the (.+?) of this paper")


class FakeKaggle:
    """Replies "<part> summary [Doc 1, Sec ...]"; fails for parts listed in ``fail``."""

    def __init__(self, fail: set[str] | None = None) -> None:
        self.fail = fail or set()
        self.calls: list[str] = []
        self._lock = threading.Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw: Any) -> Any:
        part = _PART_RE.search(kw["messages"][1]["content"]).group(1)
        with self._lock:
            self.calls.append(part)
        if part in self.fail:
            raise RuntimeError("kaggle down")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=f"{part} summary [Doc 1, Sec X]."))])


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    s = Settings(_env_file=None, finetuned_base_url="http://kaggle.test/v1",
                 llm_base_url="https://api.groq.com/openai/v1", llm_model="openai/gpt-oss-120b")
    monkeypatch.setattr(router_mod, "get_settings", lambda: s)
    return s


def _tool(settings: Settings, kaggle: FakeKaggle, fallback: Any = None) -> SummarizePapersTool:
    healthy = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"data": []})))
    router = FinetunedRouter(settings=settings, client=kaggle, http_client=healthy)
    session = Session(session_id="s", doc_titles={"d6": "R6.pdf"})
    return SummarizePapersTool(session, doc_chunks_fn=lambda doc: CHUNKS if doc == "R6.pdf" else [],
                               router=router, fallback_llm=fallback)


def test_summarize_combines_five_cited_sections(settings: Settings) -> None:
    kaggle = FakeKaggle()
    tool = _tool(settings, kaggle)

    out = tool({"doc": "R6.pdf"})

    headings = re.findall(r"^### (.+)$", out, re.M)
    assert headings == ["Objective", "Method", "Dataset/Experiments", "Results", "Conclusion"]
    assert sorted(kaggle.calls) == sorted(["objective", "method", "dataset/experiments", "results", "conclusion"])
    assert out.startswith("**Summary of R6.pdf** (Doc 1)")
    assert out.count("[Doc 1, Sec X]") == 5  # each section keeps its citation
    assert tool.model_used == "Fine-tuned Qwen (5/5 sections)"
    assert [c.chunk_id for _, c in tool.retrieved] == [c.chunk_id for c in CHUNKS]


def test_failed_section_falls_back_alone_and_label_reports_mix(settings: Settings) -> None:
    groq = SimpleNamespace(generate=lambda messages, max_tokens=512, temperature=0.0: "Groq part.")
    tool = _tool(settings, FakeKaggle(fail={"results"}), fallback=groq)

    out = tool({"doc": "R6.pdf"})

    assert "### Results\nGroq part. [Doc 1, Sec 5 Experimental Results]" in out  # citation re-attached
    assert tool.model_used.startswith("Fine-tuned Qwen (") and "Groq gpt-oss-120b" in tool.model_used


def test_sections_split_evenly_when_headings_missing() -> None:
    chunks = [Chunk(chunk_id=f"c{i}", doc_id="d", text="t", page=i, section=None) for i in range(10)]
    groups = group_sections(chunks)
    assert [h for h, _ in groups] == ["Objective", "Method", "Dataset/Experiments", "Results", "Conclusion"]
    assert [len(g) for _, g in groups] == [2, 2, 2, 2, 2]
