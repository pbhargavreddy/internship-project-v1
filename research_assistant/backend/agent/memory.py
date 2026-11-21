"""
memory.py
---------
Per-session conversation memory for the ReAct agent.

Each session is one JSON file, ``<sessions_dir>/<session_id>.json``, holding
the past turns, the stable ``Doc N`` numbering of every document the agent
has seen, and the chunk locations it can cite.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from pydantic import BaseModel, Field

from backend.config import get_settings
from backend.ingestion.names import doc_name

log = logging.getLogger(__name__)

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_SOURCES = 200


class Citation(BaseModel):
    """A citable location: ``[Doc N, Sec S]`` resolved to a title and page."""

    doc: int
    title: str
    page: int | None = None
    section: str | None = None


class Turn(BaseModel):
    """One user message and the agent's final answer."""

    user: str
    assistant: str


def count_tokens(text: str) -> int:
    """Token estimate with tiktoken ``cl100k_base`` (close enough for budgets on Groq models)."""
    import tiktoken

    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def trim_history(turns: list[Turn], max_messages: int, max_tokens: int) -> list[Turn]:
    """The newest turns that fit in ``max_messages`` messages and ``max_tokens`` tokens.

    Whole turns are dropped oldest-first. If even the newest turn alone is over the cap,
    its answer is cut so the reference the user is following up on survives.

    Args:
        turns: Past turns, oldest first.
        max_messages: Message budget; each turn is 2 messages (user + assistant).
        max_tokens: Token budget (tiktoken ``cl100k_base``) over all kept messages.

    Returns:
        The kept turns, oldest first.
    """
    n = max_messages // 2
    kept = list(turns[-n:]) if n > 0 else []
    costs = [count_tokens(t.user) + count_tokens(t.assistant) for t in kept]
    while len(kept) > 1 and sum(costs) > max_tokens:
        kept.pop(0)
        costs.pop(0)
    if kept and costs[0] > max_tokens:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        budget = max(max_tokens - count_tokens(kept[0].user), 0)
        kept[0] = Turn(user=kept[0].user, assistant=enc.decode(enc.encode(kept[0].assistant)[:budget]))
    return kept


class Session(BaseModel):
    """Everything the agent remembers about one conversation."""

    session_id: str
    turns: list[Turn] = Field(default_factory=list)
    doc_numbers: dict[str, int] = Field(default_factory=dict, description="doc_id -> N")
    doc_titles: dict[str, str] = Field(default_factory=dict, description="doc_id -> title")
    sources: list[Citation] = Field(default_factory=list, description="Chunks shown to the model")

    def doc_number(self, doc_id: str) -> int:
        """Return the stable ``N`` for ``doc_id``, assigning the next one if new."""
        if doc_id not in self.doc_numbers:
            self.doc_numbers[doc_id] = len(self.doc_numbers) + 1
        return self.doc_numbers[doc_id]

    def title_for(self, doc_id: str) -> str:
        """Title of ``doc_id``: this session's, else the shared names registry, else a placeholder.

        The registry (``data/uploads/names.json``) holds upload filenames and the titles of
        papers ingested by paper_search in any session.
        """
        # session_id is the chat_id, so a chat shows the filename it uploaded the document under.
        return self.doc_titles.get(doc_id) or doc_name(doc_id, chat_id=self.session_id) or f"Document {doc_id}"

    def doc_id_for(self, number: int) -> str | None:
        """Inverse of :meth:`doc_number`; ``None`` if ``number`` was never assigned."""
        return next((d for d, n in self.doc_numbers.items() if n == number), None)

    def add_source(self, citation: Citation) -> None:
        """Remember a citable chunk location (deduplicated, capped)."""
        if citation not in self.sources:
            self.sources.append(citation)
            del self.sources[:-MAX_SOURCES]


class SessionStore:
    """Loads and saves :class:`Session` objects as JSON files.

    Args:
        root: Directory for session files. Defaults to ``settings.sessions_dir``.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root: Path = Path(root) if root is not None else get_settings().sessions_dir

    def _path(self, session_id: str) -> Path:
        if not SESSION_ID_RE.match(session_id):
            raise ValueError("session_id must be 1-64 characters of letters, digits, '_' or '-'")
        return self.root / f"{session_id}.json"

    def load(self, session_id: str) -> Session:
        """Return the saved session, or a new empty one.

        Raises:
            ValueError: If ``session_id`` contains disallowed characters.
        """
        path = self._path(session_id)
        if not path.exists():
            return Session(session_id=session_id)
        try:
            return Session.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            log.warning("Corrupt session file %s; starting fresh", path)
            return Session(session_id=session_id)

    def delete(self, session_id: str) -> None:
        """Remove a session file (no error if it doesn't exist)."""
        self._path(session_id).unlink(missing_ok=True)

    def save(self, session: Session) -> None:
        """Write the session atomically (temp file, then rename)."""
        path = self._path(session.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(session.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)
