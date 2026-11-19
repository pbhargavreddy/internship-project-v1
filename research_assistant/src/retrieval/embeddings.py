"""
embeddings.py
-------------
SentenceTransformer embedding wrapper providing a clean interface for
embedding lists of documents and single query strings.
"""

from __future__ import annotations

from typing import List

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Thin wrapper around a ``sentence-transformers`` model.

    By default uses ``all-MiniLM-L6-v2`` (fast, 384-dim) but can be
    swapped for the more accurate ``all-mpnet-base-v2`` (768-dim) by
    passing the model name at construction.

    Args:
        model_name: HuggingFace model identifier.
        device:     ``"cpu"``, ``"cuda"``, or ``None`` (auto-detect).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document strings.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            List of float vectors, one per input string.

        Raises:
            ValueError: If ``texts`` is empty.
        """
        if not texts:
            raise ValueError("texts must be a non-empty list")
        embeddings = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,  # unit-normalised → cosine == dot product
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string.

        Args:
            text: Query string to embed.

        Returns:
            Float vector representing the query.

        Raises:
            ValueError: If ``text`` is empty or whitespace.
        """
        if not text or not text.strip():
            raise ValueError("Query text must not be empty")
        embedding = self._model.encode(
            text,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embedding.tolist()

    @property
    def embedding_dimension(self) -> int:
        """Return the output dimensionality of the loaded model."""
        return self._model.get_sentence_embedding_dimension()

