"""
faithfulness.py
---------------
Embedding-based faithfulness check for agent answers.

The answer is split into sentences; each sentence is "supported" when its
best cosine similarity to any cited chunk is at least
``settings.faithfulness_threshold`` (bge embeddings, normalized). The score is
supported / total sentences.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
from pydantic import BaseModel

from backend.config import get_settings
from backend.retrieval.vector_store import BgeEmbedder, Embedder

_CITE_RE = re.compile(r"\[Doc[^\]]*\]", re.I)
_MD_RE = re.compile(r"^\s*(#+|[-*+]|\d+[.)])\s+|[*_`>]+", re.M)
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
MIN_WORDS = 4  # shorter fragments (headings, "Yes.") aren't checked


class FaithfulnessResult(BaseModel):
    score: float
    total: int
    unsupported: list[str]


def split_sentences(answer: str) -> list[str]:
    """Split Markdown ``answer`` into checkable sentences (citations and markup removed)."""
    text = _MD_RE.sub(" ", _CITE_RE.sub("", answer))
    parts = (re.sub(r"\s+", " ", p).strip() for p in _SPLIT_RE.split(text))
    return [p for p in parts if len(p.split()) >= MIN_WORDS]


@lru_cache(maxsize=1)
def _default_embedder() -> BgeEmbedder:
    return BgeEmbedder()


def check_faithfulness(
    answer: str,
    cited_chunks: list[str],
    embedder: Embedder | None = None,
    threshold: float | None = None,
) -> FaithfulnessResult:
    """Score how well ``answer``'s sentences are supported by ``cited_chunks``.

    Args:
        answer: The agent's final answer (Markdown).
        cited_chunks: Texts of the chunks the answer cites.
        embedder: Embedding backend. Defaults to the bge model from settings.
        threshold: Minimum max-cosine for a supported sentence. Defaults to ``settings.faithfulness_threshold``.

    Returns:
        Score in [0, 1] and the unsupported sentences. An answer with no
        checkable sentences scores 1.0; one with sentences but no chunks scores 0.0.
    """
    threshold = threshold if threshold is not None else get_settings().faithfulness_threshold
    sentences = split_sentences(answer)
    if not sentences:
        return FaithfulnessResult(score=1.0, total=0, unsupported=[])
    if not cited_chunks:
        return FaithfulnessResult(score=0.0, total=len(sentences), unsupported=sentences)
    emb = embedder or _default_embedder()
    sims = np.asarray(emb.embed_documents(sentences)) @ np.asarray(emb.embed_documents(cited_chunks)).T
    best = sims.max(axis=1)
    unsupported = [s for s, b in zip(sentences, best) if b < threshold]
    score = (len(sentences) - len(unsupported)) / len(sentences)
    return FaithfulnessResult(score=round(score, 3), total=len(sentences), unsupported=unsupported)
