"""
doc_store.py
------------
Durable document storage in MongoDB (the PDF files themselves are not kept):

  documents: {doc_id, chat_id, filename, title, uploaded_at, n_chunks}
  chunks:    {doc_id, chat_id, chunk_id, text, page, section, embedding}

Both collections are indexed on ``(chat_id, doc_id)``. The local ChromaDB is a
rebuildable cache of ``chunks`` (embeddings are stored, so a rebuild never
re-embeds). Every function takes the database explicitly; callers get it from
:func:`backend.storage.mongo.get_db` and skip Mongo when it is ``None``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from pydantic import BaseModel, Field
from pymongo import ASCENDING
from pymongo.database import Database

from backend.ingestion import Chunk

log = logging.getLogger(__name__)

DOCUMENTS = "documents"
CHUNKS = "chunks"


class DocumentRecord(BaseModel):
    doc_id: str
    chat_id: str
    filename: str
    title: str
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    n_chunks: int


class StoredChunk(BaseModel):
    doc_id: str
    chat_id: str
    chunk_id: str
    text: str
    page: int
    section: str | None = None
    embedding: list[float]

    def to_chunk(self) -> Chunk:
        return Chunk(chunk_id=self.chunk_id, doc_id=self.doc_id, text=self.text, page=self.page, section=self.section)


def ensure_indexes(db: Database) -> None:
    """``(chat_id, doc_id)`` indexes; one document record per pair."""
    db[DOCUMENTS].create_index([("chat_id", ASCENDING), ("doc_id", ASCENDING)], unique=True)
    db[CHUNKS].create_index([("chat_id", ASCENDING), ("doc_id", ASCENDING)])


def save_document(db: Database, record: DocumentRecord, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    """Write (or replace) one chat's copy of a document with its chunks and embeddings.

    Raises:
        ValueError: If ``chunks`` and ``embeddings`` differ in length.
    """
    if len(chunks) != len(embeddings):
        raise ValueError("one embedding per chunk is required")
    ensure_indexes(db)
    key = {"chat_id": record.chat_id, "doc_id": record.doc_id}
    db[CHUNKS].delete_many(key)
    if chunks:
        db[CHUNKS].insert_many([
            StoredChunk(doc_id=c.doc_id, chat_id=record.chat_id, chunk_id=c.chunk_id, text=c.text, page=c.page,
                        section=c.section, embedding=[float(x) for x in e]).model_dump()
            for c, e in zip(chunks, embeddings)
        ])
    db[DOCUMENTS].replace_one(key, record.model_dump(), upsert=True)


def has_document(db: Database, doc_id: str, chat_id: str) -> bool:
    return db[DOCUMENTS].count_documents({"chat_id": chat_id, "doc_id": doc_id}, limit=1) > 0


def list_documents(db: Database, chat_id: str | None = None) -> list[DocumentRecord]:
    """Document records (of one chat, if given)."""
    query = {"chat_id": chat_id} if chat_id else {}
    return [DocumentRecord(**{k: v for k, v in d.items() if k != "_id"}) for d in db[DOCUMENTS].find(query)]


def load_chunks(db: Database, doc_id: str, chat_id: str) -> list[StoredChunk]:
    """One chat's chunks of a document, in page order."""
    docs = db[CHUNKS].find({"chat_id": chat_id, "doc_id": doc_id}, {"_id": 0})
    return sorted((StoredChunk(**d) for d in docs), key=lambda c: c.page)


def delete_document(db: Database, doc_id: str, chat_id: str) -> int:
    """Remove one chat's copy of a document; returns how many chunks were deleted."""
    key = {"chat_id": chat_id, "doc_id": doc_id}
    db[DOCUMENTS].delete_many(key)
    return db[CHUNKS].delete_many(key).deleted_count
