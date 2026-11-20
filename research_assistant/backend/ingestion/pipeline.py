"""
pipeline.py
-----------
PDF → chunks → JSONL on disk.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from backend.ingestion.chunker import Chunk, chunk_blocks
from backend.ingestion.pdf_loader import load_pdf
from backend.runtime import timed

log = logging.getLogger(__name__)

DEFAULT_CHUNKS_DIR = "data/chunks"


def compute_doc_id(path: str | Path) -> str:
    """Content hash of the file, so the same PDF always gets the same ID."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def ingest_pdf(
    path: str | Path, out_dir: str | Path = DEFAULT_CHUNKS_DIR, max_pages: int | None = None
) -> list[Chunk]:
    """
    Load, chunk and persist a PDF.

    Chunks are written to ``<out_dir>/<doc_id>.jsonl`` (one Chunk per line),
    overwriting any earlier ingestion of the same file.

    Args:
        path:    PDF file to ingest.
        out_dir: Directory for the JSONL output.
        max_pages: Only ingest the first N pages (``None`` = all).

    Returns:
        The chunks that were written.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError:        If the file is not a readable PDF.
    """
    doc_id = compute_doc_id(path)
    with timed("chunk", doc_id=doc_id, max_pages=max_pages):
        chunks = chunk_blocks(load_pdf(path, max_pages=max_pages), doc_id=doc_id)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{doc_id}.jsonl"
    tmp = target.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(chunk.model_dump_json() + "\n")
    tmp.replace(target)

    log.info("Ingested %s as %s: %d chunks → %s", Path(path).name, doc_id, len(chunks), target)
    return chunks
