"""
names.py
--------
Process-wide ``doc_id -> display name`` registry, stored as JSON at
``settings.doc_names_file`` (``data/uploads/names.json``).

Uploads record their original filename and ``paper_search`` ingests record the
paper title, so citations show a real name in every session instead of
``Document <doc_id>``.

Backfill names for documents ingested before this existed:
    python -m backend.ingestion.names
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from backend.config import get_settings

log = logging.getLogger(__name__)

_lock = threading.Lock()


def _path(path: str | Path | None) -> Path:
    return Path(path) if path is not None else get_settings().doc_names_file


def load_doc_names(path: str | Path | None = None) -> dict[str, str]:
    """All known names (empty if the file is missing or unreadable).

    Args:
        path: Registry file. Defaults to ``settings.doc_names_file``.
    """
    try:
        data = json.loads(_path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def normalize_doc_name(name: str) -> str:
    """Case-insensitive key for a document name: ``" R6.PDF "`` and ``"r6"`` both give ``"r6"``."""
    key = " ".join(name.strip().casefold().split())
    return key[:-4].rstrip() if key.endswith(".pdf") else key


def is_known_doc_name(text: str, path: str | Path | None = None) -> bool:
    """Whether ``text`` is (case-insensitively) the name of a registered document."""
    key = normalize_doc_name(text)
    return bool(key) and any(normalize_doc_name(n) == key for n in load_doc_names(path).values())


def doc_name(doc_id: str, path: str | Path | None = None) -> str | None:
    """Registered name of ``doc_id``, or ``None``."""
    return load_doc_names(path).get(doc_id)


def save_doc_names(updates: dict[str, str], path: str | Path | None = None) -> None:
    """Merge ``updates`` into the registry (atomic write).

    Args:
        updates: ``doc_id -> name`` pairs; existing IDs are overwritten.
        path: Registry file. Defaults to ``settings.doc_names_file``.
    """
    if not updates:
        return
    target = _path(path)
    with _lock:
        names = load_doc_names(target) | updates
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(names, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)


def _pdf_title(path: Path) -> str | None:
    """PDF metadata title, ignoring empty / placeholder values."""
    import pymupdf

    try:
        with pymupdf.open(path) as doc:
            title = (doc.metadata or {}).get("title", "").strip()
    except Exception:  # noqa: BLE001 — unreadable PDF: no title
        return None
    return title if title and title.lower() not in {"untitled", "title"} else None


def backfill_doc_names(
    uploads_dir: str | Path = "data/uploads",
    pdfs_dir: str | Path | None = "data/pdfs",
    sessions_dir: str | Path | None = None,
    path: str | Path | None = None,
) -> dict[str, str]:
    """Name documents that have none yet; never overwrites an existing name.

    Sources, in order: titles recorded in saved sessions (paper_search ingests),
    then each uploaded / downloaded PDF's metadata title, then its file name.

    Args:
        uploads_dir: Uploaded PDFs, stored as ``<doc_id>.pdf``.
        pdfs_dir: PDFs downloaded by paper_search (any file name); ``None`` to skip.
        sessions_dir: Session JSON files. Defaults to ``settings.sessions_dir``.
        path: Registry file. Defaults to ``settings.doc_names_file``.

    Returns:
        The names that were added.
    """
    from backend.ingestion.pipeline import compute_doc_id

    known = load_doc_names(path)
    added: dict[str, str] = {}

    def add(doc_id: str, name: str | None) -> None:
        if name and doc_id not in known and doc_id not in added:
            added[doc_id] = name

    for session_file in sorted(Path(sessions_dir or get_settings().sessions_dir).glob("*.json")):
        try:
            titles = json.loads(session_file.read_text(encoding="utf-8")).get("doc_titles", {})
        except (OSError, ValueError, AttributeError):
            continue
        for doc_id, title in titles.items():
            add(doc_id, title)
    for pdf in sorted(Path(uploads_dir).glob("*.pdf")):
        add(pdf.stem, _pdf_title(pdf) or pdf.name)
    if pdfs_dir is not None:
        for pdf in sorted(Path(pdfs_dir).glob("*.pdf")):
            add(compute_doc_id(pdf), _pdf_title(pdf) or pdf.name)

    save_doc_names(added, path)
    log.info("Backfilled %d document names", len(added))
    return added


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for doc_id, name in backfill_doc_names().items():
        print(f"{doc_id}: {name}")
