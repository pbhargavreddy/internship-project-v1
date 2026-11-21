"""
names.py
--------
Document registry, stored as JSON at ``settings.doc_names_file``
(``data/uploads/names.json``)::

    {"<doc_id>": {"name": "R6.pdf", "chats": {"<chat_id>": "R6.pdf", ...}}}

``name`` is the default display name (upload filename or paper title); ``chats``
maps every chat the document belongs to onto the filename used in that chat.
Old files (``{"<doc_id>": "R6.pdf"}``) are still read; such documents belong to
no chat until ``backend/storage/documents.py::migrate_legacy_documents`` runs.

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

_lock = threading.RLock()

Entry = dict[str, object]  # {"name": str, "chats": {chat_id: name}}


def _path(path: str | Path | None) -> Path:
    return Path(path) if path is not None else get_settings().doc_names_file


def _entry(value: object) -> Entry:
    if isinstance(value, dict):
        chats = value.get("chats") or {}
        return {"name": str(value.get("name") or ""), "chats": {str(k): str(v) for k, v in dict(chats).items()}}
    return {"name": str(value), "chats": {}}  # legacy: plain name, no chat


def _load_raw(path: str | Path | None = None) -> dict[str, Entry]:
    try:
        data = json.loads(_path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): _entry(v) for k, v in data.items()} if isinstance(data, dict) else {}


def _write_raw(entries: dict[str, Entry], path: str | Path | None = None) -> None:
    target = _path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)


def load_doc_names(path: str | Path | None = None, chat_id: str | None = None) -> dict[str, str]:
    """``doc_id -> name`` (empty if the file is missing or unreadable).

    Args:
        path: Registry file. Defaults to ``settings.doc_names_file``.
        chat_id: Only documents of this chat, with the filename used there.
    """
    out: dict[str, str] = {}
    for doc_id, e in _load_raw(path).items():
        chats: dict[str, str] = e["chats"]  # type: ignore[assignment]
        if chat_id is None:
            out[doc_id] = str(e["name"]) or next(iter(chats.values()), "")
        elif chat_id in chats:
            out[doc_id] = chats[chat_id] or str(e["name"])
    return out


def doc_chats(doc_id: str, path: str | Path | None = None) -> list[str]:
    """Chats that ``doc_id`` belongs to."""
    e = _load_raw(path).get(doc_id)
    return list(e["chats"]) if e else []  # type: ignore[arg-type]


def legacy_doc_ids(path: str | Path | None = None) -> list[str]:
    """Registered documents that belong to no chat (pre-migration entries)."""
    return [d for d, e in _load_raw(path).items() if not e["chats"]]


def normalize_doc_name(name: str) -> str:
    """Case-insensitive key for a document name: ``" R6.PDF "`` and ``"r6"`` both give ``"r6"``."""
    key = " ".join(name.strip().casefold().split())
    return key[:-4].rstrip() if key.endswith(".pdf") else key


def is_known_doc_name(text: str, path: str | Path | None = None, chat_id: str | None = None) -> bool:
    """Whether ``text`` is (case-insensitively) the name of a registered document (of ``chat_id``)."""
    key = normalize_doc_name(text)
    return bool(key) and any(normalize_doc_name(n) == key for n in load_doc_names(path, chat_id).values())


def doc_name(doc_id: str, path: str | Path | None = None, chat_id: str | None = None) -> str | None:
    """Name of ``doc_id``: the filename used in ``chat_id`` if given and known, else the default name."""
    e = _load_raw(path).get(doc_id)
    if e is None:
        return None
    chats: dict[str, str] = e["chats"]  # type: ignore[assignment]
    return (chats.get(chat_id) if chat_id else None) or str(e["name"]) or next(iter(chats.values()), None)


def save_doc_names(updates: dict[str, str], path: str | Path | None = None) -> None:
    """Set default names (chat membership is kept).

    Args:
        updates: ``doc_id -> name`` pairs; existing names are overwritten.
        path: Registry file. Defaults to ``settings.doc_names_file``.
    """
    if not updates:
        return
    with _lock:
        entries = _load_raw(path)
        for doc_id, name in updates.items():
            entries.setdefault(doc_id, {"name": "", "chats": {}})["name"] = name
        _write_raw(entries, path)


def attach_doc(doc_id: str, name: str, chat_id: str, path: str | Path | None = None) -> None:
    """Record that ``doc_id`` belongs to ``chat_id`` under ``name`` (the default name is kept if set)."""
    with _lock:
        entries = _load_raw(path)
        e = entries.setdefault(doc_id, {"name": name, "chats": {}})
        e["name"] = e["name"] or name
        e["chats"][chat_id] = name  # type: ignore[index]
        _write_raw(entries, path)


def detach_doc(doc_id: str, chat_id: str, path: str | Path | None = None) -> list[str]:
    """Remove ``chat_id`` from ``doc_id``; the entry is dropped once no chat is left.

    Returns:
        The chats that still reference the document.
    """
    with _lock:
        entries = _load_raw(path)
        e = entries.get(doc_id)
        if e is None:
            return []
        chats: dict[str, str] = e["chats"]  # type: ignore[assignment]
        chats.pop(chat_id, None)
        if not chats:
            del entries[doc_id]
        _write_raw(entries, path)
        return list(chats)


def remove_doc_names(doc_ids: list[str], path: str | Path | None = None) -> None:
    """Drop ``doc_ids`` from the registry entirely (missing IDs are ignored)."""
    with _lock:
        entries = _load_raw(path)
        if not any(d in entries for d in doc_ids):
            return
        for d in doc_ids:
            entries.pop(d, None)
        _write_raw(entries, path)


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
