"""
chunker.py
----------
Token-based sliding-window chunking with tiktoken.

Chunks never cross section boundaries; each chunk records the page its first
token came from.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from itertools import groupby

import tiktoken
from pydantic import BaseModel

from backend.ingestion.pdf_loader import TextBlock

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
ENCODING_NAME = "cl100k_base"


class Chunk(BaseModel):
    """One retrievable piece of a document."""

    chunk_id: str
    doc_id: str
    text: str
    page: int
    section: str | None


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING_NAME)


def make_chunk_id(doc_id: str, index: int) -> str:
    """Deterministic ID: sha256(doc_id + index), truncated."""
    return hashlib.sha256(f"{doc_id}:{index}".encode()).hexdigest()[:16]


def chunk_blocks(
    blocks: list[TextBlock],
    doc_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    Split text blocks into overlapping token windows.

    Args:
        blocks:        Output of :func:`load_pdf`.
        doc_id:        Document identifier stored on every chunk.
        chunk_size:    Maximum tokens per chunk.
        chunk_overlap: Tokens shared by consecutive chunks of one section.

    Returns:
        Chunks in document order with globally sequential IDs.

    Raises:
        ValueError: If ``chunk_overlap`` is not smaller than ``chunk_size``.
    """
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
    enc = _encoding()
    step = chunk_size - chunk_overlap
    chunks: list[Chunk] = []

    for section, group in groupby(blocks, key=lambda b: b.section):
        tokens: list[int] = []
        pages: list[int] = []
        for block in group:
            block_tokens = enc.encode((" " if tokens else "") + block.text)
            tokens.extend(block_tokens)
            pages.extend([block.page] * len(block_tokens))

        for start in range(0, len(tokens), step):
            text = enc.decode(tokens[start : start + chunk_size]).strip()
            if text:
                chunks.append(
                    Chunk(
                        chunk_id=make_chunk_id(doc_id, len(chunks)),
                        doc_id=doc_id,
                        text=text,
                        page=pages[start],
                        section=section,
                    )
                )
            if start + chunk_size >= len(tokens):
                break
    return chunks
