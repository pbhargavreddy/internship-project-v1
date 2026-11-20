"""
test_vector_store.py
--------------------
Offline tests for backend/retrieval/vector_store.py. A deterministic
bag-of-words embedder replaces bge so no model download is needed.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import pytest

from backend.ingestion import Chunk
from backend.retrieval import VectorStore

DIM = 64


class FakeEmbedder:
    """Hashes words into a normalized bag-of-words vector."""

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for word in re.findall(r"[a-z]+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def _chunk(cid: str, doc_id: str, text: str, section: str | None = "Intro") -> Chunk:
    return Chunk(chunk_id=cid, doc_id=doc_id, text=text, page=1, section=section)


CHUNKS = [
    _chunk("a1", "docA", "transformers use self attention over token sequences"),
    _chunk("a2", "docA", "convolutional networks classify images with filters", section=None),
    _chunk("b1", "docB", "attention mechanisms improve machine translation quality"),
]


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    vs = VectorStore(persist_dir=tmp_path / "chroma", collection_name="test", embedder=FakeEmbedder())
    vs.add_chunks(CHUNKS)
    return vs


def test_add_and_search_returns_relevant_chunk(store: VectorStore) -> None:
    hits = store.search("convolutional image filters", k=2)
    top, score = hits[0]
    assert top.chunk_id == "a2"
    assert top.section is None and top.page == 1 and top.doc_id == "docA"
    assert score > hits[1][1]


def test_readding_does_not_duplicate(store: VectorStore) -> None:
    store.add_chunks(CHUNKS)
    assert store.count() == len(CHUNKS)


def test_doc_id_filter(store: VectorStore) -> None:
    hits = store.search("attention", k=5, doc_ids=["docB"])
    assert [c.chunk_id for c, _ in hits] == ["b1"]


def test_get_doc_chunks_by_filename_or_doc_id_in_page_order(tmp_path: Path, monkeypatch) -> None:
    import backend.ingestion.names as names
    from backend.config import Settings

    monkeypatch.setattr(names, "get_settings", lambda: Settings(_env_file=None, doc_names_file=tmp_path / "names.json"))
    names.save_doc_names({"d6": "R6.pdf", "d7": "R7.pdf"})
    vs = VectorStore(persist_dir=tmp_path / "chroma", collection_name="docs", embedder=FakeEmbedder())
    vs.add_chunks([
        Chunk(chunk_id="p3", doc_id="d6", text="results table", page=3, section="Results"),
        Chunk(chunk_id="p1", doc_id="d6", text="intro text", page=1, section="Introduction"),
        Chunk(chunk_id="x1", doc_id="d7", text="other paper", page=1, section="Introduction"),
    ])

    for ref in ["r6", "R6.pdf", " r6.PDF ", "d6"]:
        assert [c.chunk_id for c in vs.get_doc_chunks(ref)] == ["p1", "p3"], ref
    assert vs.get_doc_chunks("R8.pdf") == []
