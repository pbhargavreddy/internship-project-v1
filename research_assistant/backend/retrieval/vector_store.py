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
        chunk_id=cid,
        doc_id=str(meta["doc_id"]),
        text=text,
        page=int(meta["page"]),
        section=str(meta["section"]) or None,  # stored as "" when missing
    )


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

    def has_doc(self, doc_id: str) -> bool:
        """Whether any chunk of ``doc_id`` is already stored."""
        return bool(self._collection.get(where={"doc_id": doc_id}, limit=1, include=[])["ids"])

    def resolve_doc_id(self, doc: str) -> str | None:
        """Map a doc_id or a document name to a stored doc_id.

        Names come from the shared registry (``data/uploads/names.json``) and match
        case-insensitively, with or without ``.pdf``: ``"r6"``, ``"R6.pdf"``.

        Args:
            doc: doc_id, upload filename or paper title.

        Returns:
            The doc_id, or ``None`` if nothing stored matches.
        """
        ref = doc.strip()
        if not ref:
            return None
        if self.has_doc(ref):
            return ref
        key = normalize_doc_name(ref)
        matches = [d for d, name in load_doc_names().items() if normalize_doc_name(name) == key and self.has_doc(d)]
        if len(matches) > 1:
            log.warning("%d documents are named %r; using the most recently registered", len(matches), doc)
        return matches[-1] if matches else None

    def get_doc_chunks(self, doc: str) -> list[Chunk]:
        """All chunks of one document in page order (metadata filter, no semantic search).

        Args:
            doc: doc_id or document name (see :meth:`resolve_doc_id`).

        Returns:
            The chunks sorted by page (insertion order within a page), or ``[]`` if unknown.
        """
        doc_id = self.resolve_doc_id(doc)
        if doc_id is None:
            return []
        res = self._collection.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
        chunks = [_chunk_from(cid, text, meta) for cid, text, meta in zip(res["ids"], res["documents"], res["metadatas"])]
        return sorted(chunks, key=lambda c: c.page)  # stable: keeps chunk order within a page

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Embed and upsert chunks by ``chunk_id``.

        Args:
            chunks: Chunks to store. Existing IDs are overwritten.

        Returns:
            Number of chunks upserted.
        """
        if not chunks:
            return 0
        with timed("embed", chunks=len(chunks)):
            embeddings = self.embedder.embed_documents([c.text for c in chunks])
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            # Chroma metadata can't hold None, so a missing section is stored as "".
            metadatas=[
                {"doc_id": c.doc_id, "page": c.page, "section": c.section or ""} for c in chunks
            ],
        )
        log.info("Upserted %d chunks into %s", len(chunks), self._collection.name)
        return len(chunks)

    def search(
        self, query: str, k: int = 5, doc_ids: list[str] | None = None
    ) -> list[tuple[Chunk, float]]:
        """Return the ``k`` chunks most similar to ``query``.

        Args:
            query: Natural-language query (the bge prefix is added by the embedder).
            k: Maximum number of results.
            doc_ids: If given, only search chunks from these documents.

        Returns:
            ``(chunk, score)`` pairs, best first; score is cosine similarity.
        """
        if k < 1 or self.count() == 0:
            return []
        where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
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
