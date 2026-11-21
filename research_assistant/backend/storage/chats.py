"""
chats.py
--------
Named chats: one document per chat in the MongoDB collection ``chats``
(or one JSON file per chat under ``settings.chats_dir`` when MongoDB is
unreachable)::

    {chat_id, name, created_at, updated_at,
     messages: [{role: "user"|"agent", content, ts, model_used, citations,
                 papers, images, steps, ...}]}

The agent's prompt history for a chat comes from its stored messages via
:class:`ChatHistorySessionStore`, so the agent itself is unchanged.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal, Protocol

from pydantic import AfterValidator, BaseModel, Field, field_validator
from pymongo import DESCENDING
from pymongo.collection import Collection

from backend.agent.memory import Citation, Session, SessionStore, Turn
from backend.agent.react import AgentNote, AgentResult, AgentStep
from backend.agent.tools import PaperRef
from backend.config import get_settings
from backend.eval.faithfulness import FaithfulnessResult
from backend.storage.mongo import get_db

log = logging.getLogger(__name__)

COLLECTION = "chats"
NAME_MAX_CHARS = 60
STEP_OBS_CHARS = 500


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    """Stored datetimes come back naive from some clients; they are always UTC."""
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


UtcDatetime = Annotated[datetime, AfterValidator(_as_utc)]


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


def validate_name(name: str) -> str:
    """Strip and check a chat name.

    Raises:
        ValueError: If the name is empty or longer than 60 characters.
    """
    name = " ".join(name.split())
    if not name:
        raise ValueError("chat name is required")
    if len(name) > NAME_MAX_CHARS:
        raise ValueError(f"chat name must be at most {NAME_MAX_CHARS} characters")
    return name


class ChatMessage(BaseModel):
    """One stored message; agent messages carry everything the UI needs to re-render them."""

    role: Literal["user", "agent"]
    content: str
    ts: UtcDatetime = Field(default_factory=_now)
    model_used: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    papers: list[PaperRef] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    steps: list[AgentStep] = Field(default_factory=list)
    notes: list[AgentNote] = Field(default_factory=list)  # agent log: re-asks, final thought, routing
    latency_ms: float | None = None
    fallback_reason: str | None = None
    faithfulness: FaithfulnessResult | None = None

    @field_validator("steps")
    @classmethod
    def _truncate_observations(cls, steps: list[AgentStep]) -> list[AgentStep]:
        return [
            s if len(s.observation) <= STEP_OBS_CHARS else s.model_copy(update={"observation": s.observation[:STEP_OBS_CHARS]})
            for s in steps
        ]

    @classmethod
    def from_result(cls, result: AgentResult) -> ChatMessage:
        """The agent message for one ``/agent/chat`` result."""
        return cls(
            role="agent", content=result.answer, model_used=result.model_used or None,
            citations=result.citations, papers=result.papers, images=result.images, steps=result.steps,
            notes=result.notes,
            latency_ms=result.latency_ms, fallback_reason=result.fallback_reason, faithfulness=result.faithfulness,
        )


class ChatSummary(BaseModel):
    chat_id: str
    name: str
    created_at: UtcDatetime
    updated_at: UtcDatetime


class Chat(ChatSummary):
    messages: list[ChatMessage] = Field(default_factory=list)


class ChatStore(Protocol):
    def create(self, name: str) -> ChatSummary: ...
    def list(self) -> list[ChatSummary]: ...
    def get(self, chat_id: str) -> Chat | None: ...
    def append(self, chat_id: str, msg: ChatMessage) -> bool: ...
    def rename(self, chat_id: str, name: str) -> ChatSummary | None: ...
    def delete(self, chat_id: str) -> bool: ...


class MongoChatStore:
    """Chats in a MongoDB collection (one document per chat).

    Args:
        collection: The ``chats`` collection.
    """

    def __init__(self, collection: Collection) -> None:
        self.col = collection
        self.col.create_index("chat_id", unique=True)
        self.col.create_index([("updated_at", DESCENDING)])

    def create(self, name: str) -> ChatSummary:
        now = _now()
        chat = Chat(chat_id=_new_id(), name=validate_name(name), created_at=now, updated_at=now)
        self.col.insert_one(chat.model_dump())
        return ChatSummary(**chat.model_dump(exclude={"messages"}))

    def list(self) -> list[ChatSummary]:
        docs = self.col.find({}, {"_id": 0, "messages": 0}).sort([("updated_at", DESCENDING), ("created_at", DESCENDING)])
        return [ChatSummary(**d) for d in docs]

    def get(self, chat_id: str) -> Chat | None:
        doc = self.col.find_one({"chat_id": chat_id}, {"_id": 0})
        return Chat(**doc) if doc else None

    def append(self, chat_id: str, msg: ChatMessage) -> bool:
        res = self.col.update_one(
            {"chat_id": chat_id}, {"$push": {"messages": msg.model_dump()}, "$set": {"updated_at": _now()}}
        )
        return res.matched_count == 1

    def rename(self, chat_id: str, name: str) -> ChatSummary | None:
        res = self.col.update_one({"chat_id": chat_id}, {"$set": {"name": validate_name(name), "updated_at": _now()}})
        if res.matched_count == 0:
            return None
        doc = self.col.find_one({"chat_id": chat_id}, {"_id": 0, "messages": 0})
        return ChatSummary(**doc)

    def delete(self, chat_id: str) -> bool:
        return self.col.delete_one({"chat_id": chat_id}).deleted_count == 1


class LocalChatStore:
    """Offline fallback: one ``<chat_id>.json`` per chat under ``root``.

    Args:
        root: Directory for chat files. Defaults to ``settings.chats_dir``.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or get_settings().chats_dir)
        self._lock = threading.Lock()

    def _path(self, chat_id: str) -> Path:
        if not chat_id.isalnum():
            raise ValueError(f"Invalid chat_id: {chat_id!r}")
        return self.root / f"{chat_id}.json"

    def _write(self, chat: Chat) -> None:
        path = self._path(chat.chat_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(chat.model_dump_json(), encoding="utf-8")
        tmp.replace(path)

    def create(self, name: str) -> ChatSummary:
        now = _now()
        chat = Chat(chat_id=_new_id(), name=validate_name(name), created_at=now, updated_at=now)
        with self._lock:
            self._write(chat)
        return ChatSummary(**chat.model_dump(exclude={"messages"}))

    def list(self) -> list[ChatSummary]:
        chats = []
        for path in self.root.glob("*.json"):
            try:
                chats.append(ChatSummary(**Chat.model_validate_json(path.read_text(encoding="utf-8")).model_dump(exclude={"messages"})))
            except (OSError, ValueError):
                log.warning("Skipping unreadable chat file %s", path.name)
        return sorted(chats, key=lambda c: (c.updated_at, c.created_at), reverse=True)

    def get(self, chat_id: str) -> Chat | None:
        try:
            return Chat.model_validate_json(self._path(chat_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def append(self, chat_id: str, msg: ChatMessage) -> bool:
        with self._lock:
            chat = self.get(chat_id)
            if chat is None:
                return False
            chat.messages.append(msg)
            chat.updated_at = _now()
            self._write(chat)
        return True

    def rename(self, chat_id: str, name: str) -> ChatSummary | None:
        name = validate_name(name)
        with self._lock:
            chat = self.get(chat_id)
            if chat is None:
                return None
            chat.name, chat.updated_at = name, _now()
            self._write(chat)
        return ChatSummary(**chat.model_dump(exclude={"messages"}))

    def delete(self, chat_id: str) -> bool:
        with self._lock:
            path = self._path(chat_id)
            existed = path.exists()
            path.unlink(missing_ok=True)
        return existed


_store: ChatStore | None = None
_store_lock = threading.Lock()


def get_chat_store() -> ChatStore:
    """MongoDB-backed store if reachable, else :class:`LocalChatStore` (decided once)."""
    global _store
    with _store_lock:
        if _store is None:
            db = get_db()
            _store = MongoChatStore(db[COLLECTION]) if db is not None else LocalChatStore()
        return _store


def set_chat_store(store: ChatStore | None) -> None:
    """Override the store (tests); ``None`` re-detects on next use."""
    global _store
    with _store_lock:
        _store = store


def is_offline() -> bool:
    """True when chats are being saved locally instead of in MongoDB."""
    return isinstance(get_chat_store(), LocalChatStore)


def create_chat(name: str) -> ChatSummary:
    """Create an empty chat.

    Raises:
        ValueError: If the name is empty or longer than 60 characters.
    """
    return get_chat_store().create(name)


def list_chats() -> list[ChatSummary]:
    """All chats, most recently updated first, without messages."""
    return get_chat_store().list()


def get_chat(chat_id: str) -> Chat | None:
    """One chat with its messages, or ``None``."""
    return get_chat_store().get(chat_id)


def append_message(chat_id: str, msg: ChatMessage) -> bool:
    """Append ``msg`` to the chat; ``False`` if the chat doesn't exist."""
    return get_chat_store().append(chat_id, msg)


def rename_chat(chat_id: str, name: str) -> ChatSummary | None:
    """Rename a chat; ``None`` if it doesn't exist.

    Raises:
        ValueError: If the name is invalid.
    """
    return get_chat_store().rename(chat_id, name)


def delete_chat(chat_id: str) -> bool:
    """Delete a chat; ``False`` if it didn't exist."""
    return get_chat_store().delete(chat_id)


def history_turns(chat: Chat, n: int) -> list[Turn]:
    """The last ``n`` complete (user, agent) exchanges of ``chat``; unanswered user messages are skipped."""
    turns: list[Turn] = []
    pending: str | None = None
    for msg in chat.messages:
        if msg.role == "user":
            pending = msg.content
        elif pending is not None:
            turns.append(Turn(user=pending, assistant=msg.content))
            pending = None
    return turns[-n:] if n > 0 else []


class ChatHistorySessionStore(SessionStore):
    """Session store whose prompt history comes from a stored chat, not the session file.

    Doc numbering and citable sources still persist in ``<sessions_dir>/<chat_id>.json``.

    Args:
        turns: History to give the agent (e.g. :func:`history_turns`).
        root: Sessions directory. Defaults to ``settings.sessions_dir``.
    """

    def __init__(self, turns: list[Turn], root: str | Path | None = None) -> None:
        super().__init__(root)
        self._turns = turns

    def load(self, session_id: str) -> Session:
        session = super().load(session_id)
        session.turns = list(self._turns)
        return session
