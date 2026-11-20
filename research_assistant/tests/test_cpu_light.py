"""
test_cpu_light.py
-----------------
Offline tests for the CPU-light serving settings: thread caps, the disabled
local LLM, the quantized one-batch reranker, capped paper ingestion and
per-step timings.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from torch import nn
from transformers import BertConfig, BertModel, BertTokenizer

import backend.ingestion as ingestion
import backend.retrieval as retrieval
import backend.tools.web_search as web_search
from backend.agent import ReActAgent, Session, SessionStore
from backend.agent.tools import PaperSearchArgs, PaperSearchTool, RagSearchTool
from backend.config import Settings
from backend.ingestion import Chunk
from backend.llm.router import FinetunedRouter
from backend.retrieval.reranker import CrossEncoder, quantize_for_cpu
from backend.runtime import limit_cpu_threads


def test_limit_cpu_threads_sets_env_and_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    before = torch.get_num_threads()
    try:
        limit_cpu_threads(2)
        assert os.environ["OMP_NUM_THREADS"] == os.environ["MKL_NUM_THREADS"] == "2"
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(before)


def test_local_llm_provider_is_rejected_and_never_imported() -> None:
    with pytest.raises(ValidationError, match="LLM_PROVIDER=local is disabled"):
        Settings(_env_file=None, llm_provider="local")
    assert Settings(_env_file=None).cpu_threads == 4
    assert "backend.llm.local_model" not in sys.modules


def test_quantized_reranker_scores_all_pairs_in_one_batch(tmp_path: Path) -> None:
    words = "attention transformer image filter network query".split()
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]))
    config = BertConfig(vocab_size=len(words) + 5, hidden_size=32, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=64)
    model = quantize_for_cpu(CrossEncoder(model_name="tiny", encoder=BertModel(config),
                                          tokenizer=BertTokenizer(str(vocab)), max_length=256))
    assert not model.training
    assert not any(type(m) is nn.Linear for m in model.modules())  # all linears quantized

    calls = []
    model.register_forward_hook(lambda _m, inp, _out: calls.append(1))
    chunks = [Chunk(chunk_id=f"c{i}", doc_id="d", text=t, page=1, section=None) for i, t in enumerate(words)]
    ranked = model.rerank("attention", chunks, top_k=3)

    assert len(calls) == 1 and len(ranked) == 3
    assert [s for _, s in ranked] == sorted((s for _, s in ranked), reverse=True)


class _FakeStore:
    def __init__(self, known: set[str]) -> None:
        self.known = known
        self.added: list[str] = []

    def has_doc(self, doc_id: str) -> bool:
        return doc_id in self.known

    def add_chunks(self, chunks: list[Chunk]) -> int:
        self.added += [c.doc_id for c in chunks]
        return len(chunks)


def test_paper_search_ingests_two_papers_first_pages_and_skips_known(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    papers = [web_search.Paper(id=f"arxiv:{i}", title=f"P{i}", url="u", pdf_url="p", source="arxiv")
              for i in range(4)]
    store = _FakeStore(known={"doc0"})
    ingested: list[tuple[str, int | None]] = []

    def fake_ingest(path: str, max_pages: int | None = None) -> list[Chunk]:
        ingested.append((path, max_pages))
        return [Chunk(chunk_id="c", doc_id=f"doc{path[-1]}", text="t", page=1, section=None)]

    monkeypatch.setattr(web_search, "search_papers", lambda q, max_results: papers)
    monkeypatch.setattr(web_search, "download_pdf", lambda p: f"/pdfs/{p.id[-1]}")
    monkeypatch.setattr(ingestion, "compute_doc_id", lambda path: f"doc{path[-1]}")
    monkeypatch.setattr(ingestion, "ingest_pdf", fake_ingest)
    monkeypatch.setattr(retrieval, "get_vector_store", lambda: store)

    out = PaperSearchTool(Session(session_id="s")).run(PaperSearchArgs(query="q", ingest=True))

    assert ingested == [("/pdfs/1", 15)]  # paper 0 already indexed; papers 2-3 over the cap
    assert store.added == ["doc1"]
    assert "Doc 1: P0" in out and "Doc 2: P1" in out and "P2;" not in out


def test_agent_steps_report_total_ms(tmp_path: Path) -> None:
    class FakeLLM:
        replies = iter([
            "Thought: look\nAction: " + json.dumps({"tool": "rag_search", "args": {"query": "x"}}),
            "Thought: done\nFinal Answer: ok",
        ])

        def generate(self, messages: list[dict[str, str]], max_tokens: int = 512, temperature: float = 0.0) -> str:
            return next(self.replies)

    hit = Chunk(chunk_id="c1", doc_id="d1", text="text", page=1, section=None)
    agent = ReActAgent(llm=FakeLLM(), store=SessionStore(tmp_path),
                       tool_factory=lambda s: [RagSearchTool(s, retrieve_fn=lambda q, k: [(hit, 1.0)])],
                       router=FinetunedRouter(settings=Settings(_env_file=None)))

    result = agent.chat("s1", "question")

    assert len(result.steps) == 1 and result.steps[0].total_ms >= 0.0
    assert "total_ms" in result.steps[0].model_dump()
