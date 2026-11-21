"""
search.py
---------
Two-stage retrieval entry point: ChromaDB vector search, then optional
cross-encoder reranking. Shared by ``GET /rag/search`` and the agent's
``rag_search`` tool.
"""

from __future__ import annotations

import logging

from backend.config import get_settings
from backend.ingestion import Chunk
from backend.retrieval.vector_store import get_vector_store

log = logging.getLogger(__name__)


def retrieve(
    query: str, k: int = 5, doc_ids: list[str] | None = None, rerank: bool = True, chat_id: str | None = None
) -> list[tuple[Chunk, float]]:
    """Return the top-``k`` chunks for ``query``.

    With ``rerank``, fetch ``rerank_candidates`` chunks from ChromaDB and
    reorder them with the cross-encoder; scores are then reranker logits, not
    cosine similarities. Falls back to vector order if no model is trained.

    Args:
        query: Natural-language query.
        k: Number of chunks to return.
        doc_ids: If given, only search these documents.
        rerank: Rerank vector hits with the trained cross-encoder.
        chat_id: If given, only search this chat's documents.

    Returns:
        ``(chunk, score)`` pairs, best first.
    """
    reranker = None
    if rerank:
        from backend.retrieval.reranker import get_reranker  # heavy: torch + transformers

        reranker = get_reranker()
        if reranker is None:
            log.warning("No trained reranker at %s; returning vector-search order", get_settings().reranker_path)
    store = get_vector_store()
    if reranker is None:
        return store.search(query, k=k, doc_ids=doc_ids, chat_id=chat_id)
    candidates = store.search(query, k=max(k, get_settings().rerank_candidates), doc_ids=doc_ids, chat_id=chat_id)
    return reranker.rerank(query, [chunk for chunk, _ in candidates], top_k=k)
