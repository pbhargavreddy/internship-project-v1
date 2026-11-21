"""
vector_store.py
---------------
Dense retrieval over ingested chunks: bge embeddings + a persistent ChromaDB
collection (cosine distance).

Chunks are upserted by ``chunk_id`` so re-ingesting a document never creates
duplicates. Queries get the bge instruction prefix; passages do not.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.config import get_settings
from backend.ingestion import Chunk
from backend.ingestion.names import load_doc_names, normalize_doc_name
from backend.runtime import timed

log = logging.getLogger(__name__)


class Embedder(Protocol):
    """Anything that turns passages and queries into vectors."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class BgeEmbedder:
    """sentence-transformers wrapper applying the bge query-prefix convention.

    Args:
        model_name: HuggingFace model ID. Defaults to ``settings.embedding_model``.
        query_prefix: Instruction prepended to queries only.
        batch_size: Passages per encode batch. Defaults to ``settings.embedding_batch_size``.
    """

    def __init__(
        self, model_name: str | None = None, query_prefix: str | None = None, batch_size: int | None = None
    ) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, load lazily

        settings = get_settings()
        self.model_name: str = model_name or settings.embedding_model
        self.query_prefix: str = (
            settings.embedding_query_prefix if query_prefix is None else query_prefix
        )
        self.batch_size: int = batch_size or settings.embedding_batch_size
        log.info("Loading embedding model %s", self.model_name)
        self._model = SentenceTransformer(self.model_name, device="cpu")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(self.query_prefix + text, normalize_embeddings=True).tolist()


def _chunk_from(cid: str, text: str, meta: dict) -> Chunk:
    return Chunk(
        chunk_id=str(meta.get("chunk_id") or cid),  # store IDs are "<chat_id>:<chunk_id>"
        doc_id=str(meta["doc_id"]),
        text=text,
        page=int(meta["page"]),
        section=str(meta["section"]) or None,  # stored as "" when missing
    )


def _where(doc_id: str | None = None, doc_ids: list[str] | None = None, chat_id: str | None = None) -> dict | None:
    """Chroma metadata filter; Chroma needs ``$and`` for more than one condition."""
    conds: list[dict] = []
    if doc_id is not None:
        conds.append({"doc_id": doc_id})
    if doc_ids:
        conds.append({"doc_id": {"$in": doc_ids}})
    if chat_id:
        conds.append({"chat_id": chat_id})
    return None if not conds else conds[0] if len(conds) == 1 else {"$and": conds}


class VectorStore:
    """Persistent ChromaDB collection of :class:`Chunk` objects.

    Args:
        persist_dir: ChromaDB directory. Defaults to ``settings.chroma_dir``.
        collection_name: Collection name. Defaults to ``settings.chroma_collection``.
        embedder: Embedding backend. Defaults to :class:`BgeEmbedder` (loaded lazily).
    """

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        settings = get_settings()
        self.persist_dir: Path = Path(persist_dir or settings.chroma_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._embedder: Embedder | None = embedder
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        # embedding_function=None: we always pass vectors ourselves.
        self._collection = self._client.get_or_create_collection(
            name=collection_name or settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = BgeEmbedder()
        return self._embedder

    def count(self) -> int:
        """Number of stored chunks."""
        return self._collection.count()

    def has_doc(self, doc_id: str, chat_id: str | None = None) -> bool:
        """Whether any chunk of ``doc_id`` is stored (in ``chat_id``, if given)."""
        return bool(self._collection.get(where=_where(doc_id=doc_id, chat_id=chat_id), limit=1, include=[])["ids"])

    def doc_chunk_count(self, doc_id: str, chat_id: str | None = None) -> int:
        """Number of chunks of ``doc_id`` (in ``chat_id``, if given)."""
        return len(self._collection.get(where=_where(doc_id=doc_id, chat_id=chat_id), include=[])["ids"])

    def resolve_doc_id(self, doc: str, chat_id: str | None = None) -> str | None:
        """Map a doc_id or a document name to a stored doc_id.

        Names come from the registry (``data/uploads/names.json``) and match
        case-insensitively, with or without ``.pdf``: ``"r6"``, ``"R6.pdf"``.
        With ``chat_id`` only that chat's documents (and their names) are considered.

        Args:
            doc: doc_id, upload filename or paper title.
            chat_id: Chat whose documents to search.

        Returns:
            The doc_id, or ``None`` if nothing stored matches.
        """
        ref = doc.strip()
        if not ref:
            return None
        if self.has_doc(ref, chat_id):
            return ref
        key = normalize_doc_name(ref)
        matches = [d for d, name in load_doc_names(chat_id=chat_id).items()
                   if normalize_doc_name(name) == key and self.has_doc(d, chat_id)]
        if len(matches) > 1:
            log.warning("%d documents are named %r; using the most recently registered", len(matches), doc)
        return matches[-1] if matches else None

    def get_doc_chunks(self, doc: str, chat_id: str | None = None) -> list[Chunk]:
        """All chunks of one document in page order (metadata filter, no semantic search).

        Args:
            doc: doc_id or document name (see :meth:`resolve_doc_id`).
            chat_id: Only this chat's copy of the document.

        Returns:
            The chunks sorted by page (insertion order within a page), or ``[]`` if unknown.
        """
        doc_id = self.resolve_doc_id(doc, chat_id)
        if doc_id is None:
            return []
        res = self._collection.get(where=_where(doc_id=doc_id, chat_id=chat_id), include=["documents", "metadatas"])
        chunks = [_chunk_from(cid, text, meta) for cid, text, meta in zip(res["ids"], res["documents"], res["metadatas"])]
        return sorted(chunks, key=lambda c: c.page)  # stable: keeps chunk order within a page

    def delete_doc(self, doc_id: str, chat_id: str | None = None) -> int:
        """Remove the chunks of ``doc_id`` (only ``chat_id``'s copy, if given); returns how many."""
        ids = self._collection.get(where=_where(doc_id=doc_id, chat_id=chat_id), include=[])["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def chat_doc_ids(self, chat_id: str) -> list[str]:
        """doc_ids with chunks in ``chat_id``."""
        metas = self._collection.get(where={"chat_id": chat_id}, include=["metadatas"])["metadatas"]
        return list(dict.fromkeys(str(m["doc_id"]) for m in metas))

    def legacy_doc_ids(self) -> list[str]:
        """doc_ids that have chunks without a ``chat_id`` (stored before per-chat documents)."""
        metas = self._collection.get(include=["metadatas"])["metadatas"]
        return list(dict.fromkeys(str(m["doc_id"]) for m in metas if not m.get("chat_id")))

    def assign_chat(self, doc_id: str, chat_id: str) -> int:
        """Move a legacy document's unscoped chunks into ``chat_id`` (embeddings are reused).

        Returns:
            Number of chunks moved.
        """
        res = self._collection.get(where={"doc_id": doc_id}, include=["documents", "metadatas", "embeddings"])
        rows = [(cid, doc, meta, emb) for cid, doc, meta, emb in
                zip(res["ids"], res["documents"], res["metadatas"], res["embeddings"]) if not meta.get("chat_id")]
        if not rows:
            return 0
        self._collection.upsert(
            ids=[f"{chat_id}:{meta.get('chunk_id') or cid}" for cid, _, meta, _ in rows],
            documents=[doc for _, doc, _, _ in rows],
            embeddings=[list(emb) for _, _, _, emb in rows],
            metadatas=[{**meta, "chat_id": chat_id, "chunk_id": meta.get("chunk_id") or cid} for cid, _, meta, _ in rows],
        )
        self._collection.delete(ids=[cid for cid, _, _, _ in rows])
        return len(rows)

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """Passage embeddings for ``chunks`` (timed as the ``embed`` stage)."""
        if not chunks:
            return []
        with timed("embed", chunks=len(chunks)):
            return self.embedder.embed_documents([c.text for c in chunks])

    def add_embedded(self, chunks: list[Chunk], embeddings: list[list[float]], chat_id: str | None = None) -> int:
        """Upsert chunks with precomputed embeddings (e.g. a rebuild from MongoDB; no embedder call).

        Args:
            chunks: Chunks to store. Existing IDs are overwritten.
            embeddings: One vector per chunk.
            chat_id: Chat the document belongs to. The same document can be stored once per
                chat: store IDs are ``<chat_id>:<chunk_id>``. ``None`` stores it unscoped (tests).

        Returns:
            Number of chunks upserted.
        """
        if not chunks:
            return 0
        scope = {"chat_id": chat_id} if chat_id else {}
        self._collection.upsert(
            ids=[f"{chat_id}:{c.chunk_id}" if chat_id else c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            # Chroma metadata can't hold None, so a missing section is stored as "".
            metadatas=[
                {"doc_id": c.doc_id, "page": c.page, "section": c.section or "", "chunk_id": c.chunk_id, **scope}
                for c in chunks
            ],
        )
        log.info("Upserted %d chunks into %s", len(chunks), self._collection.name)
        return len(chunks)

    def add_chunks(self, chunks: list[Chunk], chat_id: str | None = None) -> int:
        """Embed and upsert chunks (see :meth:`add_embedded`)."""
        return self.add_embedded(chunks, self.embed_chunks(chunks), chat_id)

    def export_doc(self, doc_id: str, chat_id: str) -> tuple[list[Chunk], list[list[float]]]:
        """One chat's chunks of a document with their stored embeddings, in page order."""
        res = self._collection.get(where=_where(doc_id=doc_id, chat_id=chat_id),
                                   include=["documents", "metadatas", "embeddings"])
        rows = sorted(zip(res["ids"], res["documents"], res["metadatas"], res["embeddings"]),
                      key=lambda r: int(r[2]["page"]))
        return [_chunk_from(cid, text, meta) for cid, text, meta, _ in rows], [[float(x) for x in e] for *_, e in rows]

    def doc_chat_pairs(self) -> list[tuple[str, str]]:
        """Every ``(doc_id, chat_id)`` stored with a chat."""
        metas = self._collection.get(include=["metadatas"])["metadatas"]
        return list(dict.fromkeys((str(m["doc_id"]), str(m["chat_id"])) for m in metas if m.get("chat_id")))

    def search(
        self, query: str, k: int = 5, doc_ids: list[str] | None = None, chat_id: str | None = None
    ) -> list[tuple[Chunk, float]]:
        """Return the ``k`` chunks most similar to ``query``.

        Args:
            query: Natural-language query (the bge prefix is added by the embedder).
            k: Maximum number of results.
            doc_ids: If given, only search chunks from these documents.
            chat_id: If given, only search this chat's documents.

        Returns:
            ``(chunk, score)`` pairs, best first; score is cosine similarity.
        """
        if k < 1 or self.count() == 0:
            return []
        where = _where(doc_ids=doc_ids, chat_id=chat_id)
        res = self._collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[tuple[Chunk, float]] = []
        for cid, text, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            hits.append((_chunk_from(cid, text, meta), 1.0 - float(dist)))
        return hits


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Return the process-wide :class:`VectorStore` (created on first use)."""
    return VectorStore()
