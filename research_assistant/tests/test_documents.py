"""
test_documents.py
-----------------
Offline tests for per-chat documents (backend/storage/documents.py, doc_store.py
and the chat-scoped retrieval behind rag_search). ChromaDB runs in ``tmp_path``
with a bag-of-words embedder; MongoDB is mongomock.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import gridfs
import mongomock
import mongomock.gridfs
import pytest

import backend.ingestion.names as names
import backend.retrieval.reranker as reranker
import backend.retrieval.search as search
import backend.storage.chats as chats
import backend.storage.documents as documents
from backend.agent import Session
from backend.agent.tools import RagSearchTool
from backend.config import Settings
from backend.ingestion import Chunk
from backend.retrieval import VectorStore
from backend.storage import doc_store

mongomock.gridfs.enable_gridfs_integration()


class BagOfWords:
    def __init__(self) -> None:
        self.calls = 0

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * 64
        for w in re.findall(r"[a-z]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class NoEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("rebuild must not re-embed")

    def embed_query(self, text: str) -> list[float]:
        raise AssertionError("rebuild must not re-embed")


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Isolated ChromaDB, names registry and mongomock db (never the real MongoDB)."""
    monkeypatch.setattr(names, "get_settings", lambda: Settings(_env_file=None, doc_names_file=tmp_path / "names.json"))
    monkeypatch.setattr(documents, "get_db", lambda: None)
    monkeypatch.setattr(documents, "_pdf_title", lambda path: None)
    store = VectorStore(persist_dir=tmp_path / "chroma", collection_name="docs", embedder=BagOfWords())
    return {"store": store, "db": mongomock.MongoClient().db, "tmp": tmp_path}


def _add(env: dict, monkeypatch: pytest.MonkeyPatch, doc_id: str, text: str, chat_id: str, name: str = "R6.pdf") -> Path:
    """add_document with a fake PDF and a fake chunker (the real one needs a real PDF)."""
    pdf = env["tmp"] / f"upload-{doc_id}-{chat_id}.pdf"
    pdf.write_bytes(b"%PDF-1.7 " + doc_id.encode())
    chunks = [Chunk(chunk_id=f"{doc_id}-c{i}", doc_id=doc_id, text=f"{text} part {i}", page=i + 1, section="Intro")
              for i in range(2)]
    monkeypatch.setattr(documents, "extract_chunks", lambda path, d, max_pages=None: chunks)
    documents.add_document(pdf, doc_id, name, chat_id, store=env["store"], db=env["db"])
    return pdf


def test_ingest_writes_chunks_with_embeddings_to_mongo_and_deletes_the_pdf(env: dict, monkeypatch) -> None:
    doc = "a" * 16
    pdf = _add(env, monkeypatch, doc, "dehazing", "chatA")

    [record] = doc_store.list_documents(env["db"], "chatA")
    stored = doc_store.load_chunks(env["db"], doc, "chatA")
    assert (record.filename, record.n_chunks) == ("R6.pdf", 2)
    assert [c.chunk_id for c in stored] == [f"{doc}-c0", f"{doc}-c1"]
    assert all(len(c.embedding) == 64 for c in stored)
    assert env["store"].has_doc(doc, "chatA") and not pdf.exists()  # in Chroma; the PDF isn't kept
    assert set(env["db"][doc_store.CHUNKS].find_one({}, {"_id": 0})) == {
        "doc_id", "chat_id", "chunk_id", "text", "page", "section", "embedding"}


def test_rebuild_restores_chroma_from_mongo_without_embedding(env: dict, monkeypatch) -> None:
    doc = "b" * 16
    _add(env, monkeypatch, doc, "attention", "chatA")
    names.remove_doc_names([doc])  # a fresh machine: no local Chroma, no names.json
    fresh = VectorStore(persist_dir=env["tmp"] / "chroma-new", collection_name="docs", embedder=NoEmbedder())

    assert documents.rebuild_from_mongo(fresh, env["db"]) == 1

    assert [c.chunk_id for c in fresh.get_doc_chunks("r6", chat_id="chatA")] == [f"{doc}-c0", f"{doc}-c1"]
    assert names.load_doc_names(chat_id="chatA") == {doc: "R6.pdf"}
    assert documents.rebuild_from_mongo(fresh, env["db"]) == 0  # already there


def test_delete_removes_it_from_mongo_chroma_and_names(env: dict, monkeypatch) -> None:
    doc = "c" * 16
    _add(env, monkeypatch, doc, "text", "chatA")

    removal = documents.remove_document(doc, "chatA", store=env["store"], db=env["db"],
                                        uploads_dir=env["tmp"], chunks_dir=env["tmp"])

    assert removal.found and removal.chunks_removed == 2 and removal.mongo_chunks_removed == 2
    assert not env["store"].has_doc(doc) and names.load_doc_names() == {}
    assert doc_store.list_documents(env["db"]) == [] and env["db"][doc_store.CHUNKS].count_documents({}) == 0


def test_rag_search_only_sees_the_current_chats_documents(env: dict, monkeypatch) -> None:
    _add(env, monkeypatch, "d" * 16, "dehazing with transformers attention", "chatA", "A.pdf")
    _add(env, monkeypatch, "e" * 16, "dehazing with transformers attention too", "chatB", "B.pdf")
    monkeypatch.setattr(search, "get_vector_store", lambda: env["store"])
    monkeypatch.setattr(reranker, "get_reranker", lambda: None)

    out = RagSearchTool(Session(session_id="chatA"))({"query": "dehazing transformers", "k": 5})

    assert "too" not in out and "[Doc 1" in out and "[Doc 2" not in out
    assert env["store"].get_doc_chunks("b.pdf", chat_id="chatA") == []  # name lookup is scoped too


def test_file_shared_by_two_chats_survives_deleting_one(env: dict, monkeypatch) -> None:
    doc = "f" * 16
    _add(env, monkeypatch, doc, "shared", "chatA", "R6.pdf")
    _add(env, monkeypatch, doc, "shared", "chatB", "copy-of-R6.pdf")

    documents.remove_chat_documents("chatA", store=env["store"], db=env["db"],
                                    uploads_dir=env["tmp"], chunks_dir=env["tmp"])

    assert not env["store"].has_doc(doc, "chatA") and env["store"].has_doc(doc, "chatB")
    assert [r.chat_id for r in doc_store.list_documents(env["db"])] == ["chatB"]
    assert names.load_doc_names(chat_id="chatB") == {doc: "copy-of-R6.pdf"}


def test_startup_migrates_legacy_docs_to_mongo_then_drops_gridfs(env: dict) -> None:
    store, db, doc = env["store"], env["db"], "9" * 16
    names.save_doc_names({doc: "R4.pdf"})  # pre-chat registry entry
    store.add_chunks([Chunk(chunk_id="x0", doc_id=doc, text="old text", page=2, section=None)])  # no chat_id
    gridfs.GridFS(db, collection="pdfs").put(b"%PDF old", filename="R4.pdf", metadata={"doc_id": doc, "chat_ids": []})
    chats.set_chat_store(chats.MongoChatStore(mongomock.MongoClient().db[chats.COLLECTION]))
    embed_calls = store.embedder.calls
    try:
        summary = documents.startup(store, db=db)

        [chat] = chats.list_chats()
        assert chat.name == "My papers" and summary.legacy_assigned == 1 and summary.synced_to_mongo == 1
        assert summary.gridfs_dropped and "pdfs.files" not in db.list_collection_names()
        assert [c.chunk_id for c in doc_store.load_chunks(db, doc, chat.chat_id)] == ["x0"]
        assert store.embedder.calls == embed_calls  # moved with stored embeddings
        assert "My papers" in summary.text()
    finally:
        chats.set_chat_store(None)
