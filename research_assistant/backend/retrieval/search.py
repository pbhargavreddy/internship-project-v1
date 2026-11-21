"""
search.py
---------
Two-stage retrieval entry point: ChromaDB vector search, then optional
cross-encoder reranking fused with the vector order (reciprocal-rank fusion,
``RERANK_FUSION_K``). Shared by ``GET /rag/search`` and the agent's
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
    settings = get_settings()
    candidates = store.search(query, k=max(k, settings.rerank_candidates), doc_ids=doc_ids, chat_id=chat_id)
    reranked = reranker.rerank(query, [chunk for chunk, _ in candidates], top_k=len(candidates))
    if settings.rerank_fusion_k <= 0:
        return reranked[:k]
    return fuse_ranks([c for c, _ in candidates], [c for c, _ in reranked], settings.rerank_fusion_k)[:k]


def fuse_ranks(vector_order: list[Chunk], rerank_order: list[Chunk], fusion_k: int) -> list[tuple[Chunk, float]]:
    """Reciprocal-rank fusion: ``1/(k + vector_rank) + 1/(k + rerank_rank)``, best first.

    Keeps a strong vector hit that the reranker scores low (e.g. because the relevant
    sentence is past its 256-token window) from dropping out of the top-k.

    Args:
        vector_order: Candidates in vector-search order.
        rerank_order: The same candidates in reranker order.
        fusion_k: RRF constant (60 is the usual default).

    Returns:
        ``(chunk, fused score)`` pairs, best first.
    """
    vec = {c.chunk_id: i for i, c in enumerate(vector_order, 1)}
    rer = {c.chunk_id: i for i, c in enumerate(rerank_order, 1)}
    worst = len(vector_order) + 1
    scored = [(c, 1 / (fusion_k + vec[c.chunk_id]) + 1 / (fusion_k + rer.get(c.chunk_id, worst))) for c in vector_order]
    return sorted(scored, key=lambda pair: -pair[1])
