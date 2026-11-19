"""
vector_store.py
---------------
ChromaDB client handling persistent collection creation, document upsert,
and semantic similarity search.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings

from src.retrieval.embeddings import EmbeddingModel


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class SearchResult:
    """Encapsulates a single similarity search hit."""

    def __init__(self, text: str, metadata: dict, distance: float) -> None:
        self.text = text
        self.metadata = metadata
        self.distance = distance  # cosine distance: 0 (identical) → 2 (opposite)

    @property
    def cosine_similarity(self) -> float:
        """Convert ChromaDB cosine *distance* to cosine *similarity*."""
        return 1.0 - self.distance

    def __repr__(self) -> str:
        return (
            f"SearchResult(similarity={self.cosine_similarity:.4f}, "
            f"source={self.metadata.get('source')}, "
            f"page={self.metadata.get('page')}, "
            f"text={self.text[:60]!r}…)"
        )


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Persistent ChromaDB-backed vector store.

    Args:
        collection_name: Name of the ChromaDB collection.
        persist_dir:     Directory for on-disk persistence (default: ``./data/chroma_db``).
        embedding_model: :class:`EmbeddingModel` instance used to vectorise queries.
    """

    def __init__(
        self,
        collection_name: str = "research_docs",
        persist_dir: str | Path = "./data/chroma_db",
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._embedding_model = embedding_model or EmbeddingModel()

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
        )

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # use cosine distance
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(
        self,
        chunks: List[str],
        metadatas: List[dict],
        ids: List[str],
    ) -> None:
        """
        Embed and upsert text chunks into the collection.

        Args:
            chunks:    List of raw text strings to index.
            metadatas: Parallel list of metadata dicts (must match ``chunks`` length).
            ids:       Parallel list of unique string IDs for each chunk.

        Raises:
            ValueError: If the input lists have mismatched lengths or are empty.
        """
        if not chunks:
            raise ValueError("chunks must not be empty")
        if not (len(chunks) == len(metadatas) == len(ids)):
            raise ValueError("chunks, metadatas, and ids must have the same length")

        embeddings = self._embedding_model.embed_documents(chunks)

        self._collection.upsert(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def similarity_search(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Retrieve the *top_k* most similar chunks for *query_text*.

        Args:
            query_text: Natural-language query string.
            top_k:      Number of results to return (default 5).

        Returns:
            List of :class:`SearchResult` objects sorted by cosine similarity
            (highest first).

        Raises:
            ValueError: If the collection is empty.
        """
        if self._collection.count() == 0:
            raise ValueError("Collection is empty. Ingest documents before querying.")

        query_embedding = self._embedding_model.embed_query(query_text)

        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        results: List[SearchResult] = []
        for doc, meta, dist in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            results.append(SearchResult(text=doc, metadata=meta, distance=dist))

        return results

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self._collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection (useful for testing)."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

