"""
mongo.py
--------
Process-wide MongoDB connection for chats and PDF backups.

The connection is tried once (``serverSelectionTimeoutMS=3000`` plus a ping).
If ``MONGODB_URI`` is unset or the server is unreachable, :func:`get_db`
returns ``None`` and callers fall back to local storage. The URI is a secret
(it can hold a password) and is never logged.
"""

from __future__ import annotations

import logging
import threading

from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import PyMongoError

from backend.config import get_settings

log = logging.getLogger(__name__)

SERVER_SELECTION_TIMEOUT_MS = 3000

_lock = threading.Lock()
_checked = False
_db: Database | None = None


def get_db() -> Database | None:
    """The configured database, or ``None`` when MongoDB is not available (checked once)."""
    global _checked, _db
    with _lock:
        if _checked:
            return _db
        _checked = True
        settings = get_settings()
        if settings.mongodb_uri is None or not settings.mongodb_uri.get_secret_value():
            log.info("MONGODB_URI not set; chats are saved locally")
            return None
        try:
            client: MongoClient = MongoClient(
                settings.mongodb_uri.get_secret_value(),
                serverSelectionTimeoutMS=SERVER_SELECTION_TIMEOUT_MS,
                tz_aware=True,
            )
            client.admin.command("ping")
        except (PyMongoError, ValueError) as exc:  # ValueError: malformed URI
            # Only the exception type: pymongo messages can echo hosts from the URI.
            log.warning("MongoDB unreachable (%s); saving locally", type(exc).__name__)
            return None
        _db = client[settings.mongodb_db]
        log.info("Connected to MongoDB database %r", settings.mongodb_db)
        return _db


def is_online() -> bool:
    """Whether chats and PDFs are being stored in MongoDB."""
    return get_db() is not None


def set_db(db: Database | None) -> None:
    """Override the connection (tests inject a mongomock database; ``None`` = offline)."""
    global _checked, _db
    with _lock:
        _checked, _db = True, db
