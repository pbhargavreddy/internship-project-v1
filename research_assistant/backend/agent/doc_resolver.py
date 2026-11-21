"""
doc_resolver.py
---------------
Finds which of a chat's documents a free-text message names.

Matching ignores case, a ``.pdf`` ending, spaces, ``-`` and ``_``, so
"journal1", "Journal 1", "journal_1" and "journal1.pdf" all mean
``journal1.pdf``. "doc 2" / "paper 2" / "file 2" refer to the chat's
second uploaded document.
"""

from __future__ import annotations

import re

from backend.ingestion.names import load_doc_names

_SEP = r"[\s_\-]*"
_ORDINAL_RE = re.compile(r"(?<![a-z0-9])(?:doc(?:ument)?|paper|file|pdf)\s*#?\s*(\d{1,2})(?![0-9])", re.I)


def doc_key(name: str) -> str:
    """Matching key of a filename: ``"Journal_1.PDF"`` → ``"journal1"``."""
    key = name.strip().casefold()
    key = key[:-4] if key.endswith(".pdf") else key
    return re.sub(r"[\s_\-]+", "", key)


def chat_documents(chat_id: str) -> dict[str, str]:
    """``filename -> doc_id`` of ``chat_id``'s documents, in upload (registry) order."""
    out: dict[str, str] = {}
    for doc_id, name in load_doc_names(chat_id=chat_id).items():
        out.setdefault(name, doc_id)
    return out


def _name_pattern(key: str) -> re.Pattern[str]:
    """``journal1`` → letters with optional separators between them, plus an optional ``.pdf``."""
    body = _SEP.join(re.escape(ch) for ch in key)
    return re.compile(rf"(?<![a-z0-9]){body}(?:\.pdf)?(?![a-z0-9])", re.I)


def resolve_docs(text: str, chat_id: str, docs: list[str] | None = None) -> list[str]:
    """Filenames of this chat's documents named in ``text``, in the order they appear.

    Args:
        text: User message (or rewritten question).
        chat_id: Chat whose documents are candidates.
        docs: Candidate filenames in upload order. Defaults to :func:`chat_documents`.

    Returns:
        Matching filenames without duplicates; empty if none is named.
    """
    names = docs if docs is not None else list(chat_documents(chat_id))
    found: list[tuple[int, str]] = []
    remaining = text
    # Longest keys first, and matched spans blanked out, so "paper 2" doesn't also count as "paper".
    for name in sorted(names, key=lambda n: len(doc_key(n)), reverse=True):
        key = doc_key(name)
        if len(key) < 2:
            continue
        m = _name_pattern(key).search(remaining)
        if m:
            found.append((m.start(), name))
            remaining = remaining[: m.start()] + " " * (m.end() - m.start()) + remaining[m.end():]
    for m in _ORDINAL_RE.finditer(remaining):
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(names):
            found.append((m.start(), names[idx]))
    return list(dict.fromkeys(name for _, name in sorted(found)))
