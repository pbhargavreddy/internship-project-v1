"""
test_chats.py
-------------
Offline tests for backend/storage/chats.py: mongomock stands in for MongoDB,
and the local-JSON fallback is exercised with an unreachable server.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import mongomock
import pytest

import backend.storage.chats as chats
import backend.storage.mongo as mongo
from backend.agent import AgentResult, AgentStep
from backend.config import Settings
from backend.storage.chats import ChatMessage, MongoChatStore


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> MongoChatStore:
    ticks = iter(range(1000))
    start = datetime(2026, 11, 20, tzinfo=timezone.utc)
    monkeypatch.setattr(chats, "_now", lambda: start + timedelta(seconds=next(ticks)))  # strictly increasing
    s = MongoChatStore(mongomock.MongoClient().db[chats.COLLECTION])
    chats.set_chat_store(s)
    yield s
    chats.set_chat_store(None)


def test_create_list_rename_delete(store: MongoChatStore) -> None:
    a = chats.create_chat("  Dehazing   survey ")
    b = chats.create_chat("Transformers")

    assert a.name == "Dehazing survey"
    assert [c.name for c in chats.list_chats()] == ["Transformers", "Dehazing survey"]  # newest first
    assert not hasattr(chats.list_chats()[0], "messages")

    assert chats.rename_chat(a.chat_id, "Remote sensing").name == "Remote sensing"
    assert [c.chat_id for c in chats.list_chats()] == [a.chat_id, b.chat_id]  # renaming touches updated_at
    with pytest.raises(ValueError):
        chats.create_chat("x" * 61)
    with pytest.raises(ValueError):
        chats.rename_chat(a.chat_id, "   ")

    assert chats.delete_chat(b.chat_id) is True
    assert chats.delete_chat(b.chat_id) is False
    assert chats.get_chat(b.chat_id) is None and chats.rename_chat(b.chat_id, "y") is None
    assert [c.chat_id for c in chats.list_chats()] == [a.chat_id]


def test_append_message_keeps_order_and_feeds_history(store: MongoChatStore) -> None:
    chat = chats.create_chat("Q&A")
    long_obs = "o" * 900
    result = AgentResult(answer="A1", citations=[], model_used="Groq gpt-oss-120b",
                         steps=[AgentStep(thought="t", tool="rag_search", args={}, observation=long_obs)])

    assert chats.append_message(chat.chat_id, ChatMessage(role="user", content="Q1"))
    assert chats.append_message(chat.chat_id, ChatMessage.from_result(result))
    assert chats.append_message(chat.chat_id, ChatMessage(role="user", content="Q2 (unanswered)"))
    assert chats.append_message("missing", ChatMessage(role="user", content="x")) is False

    stored = chats.get_chat(chat.chat_id)
    assert [(m.role, m.content) for m in stored.messages] == [("user", "Q1"), ("agent", "A1"), ("user", "Q2 (unanswered)")]
    assert stored.messages[1].model_used == "Groq gpt-oss-120b"
    assert len(stored.messages[1].steps[0].observation) == 500
    assert [(t.user, t.assistant) for t in chats.history_turns(stored, 6)] == [("Q1", "A1")]
    assert chats.list_chats()[0].updated_at > chat.updated_at


def test_unreachable_mongo_falls_back_to_local_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    uri = "mongodb://scholar:s3cret-pw@127.0.0.1:1/?directConnection=true"
    settings = Settings(_env_file=None, mongodb_uri=uri, chats_dir=tmp_path / "chats")
    monkeypatch.setattr(mongo, "get_settings", lambda: settings)
    monkeypatch.setattr(chats, "get_settings", lambda: settings)
    monkeypatch.setattr(mongo, "SERVER_SELECTION_TIMEOUT_MS", 200)
    monkeypatch.setattr(mongo, "_checked", False)
    chats.set_chat_store(None)
    try:
        with caplog.at_level(logging.INFO):
            chat = chats.create_chat("Offline chat")
        assert chats.is_offline() and mongo.is_online() is False
        assert (tmp_path / "chats" / f"{chat.chat_id}.json").is_file()
        assert chats.append_message(chat.chat_id, ChatMessage(role="user", content="hi"))
        assert [m.content for m in chats.get_chat(chat.chat_id).messages] == ["hi"]
        assert "s3cret-pw" not in caplog.text and "127.0.0.1:1" not in caplog.text  # URI never logged
    finally:
        chats.set_chat_store(None)
        mongo.set_db(None)
        monkeypatch.setattr(mongo, "_checked", False)


def test_history_includes_papers_and_never_mixes_chats(store: MongoChatStore) -> None:
    from backend.agent.tools import PaperRef

    a, b = chats.create_chat("A"), chats.create_chat("B")
    paper = PaperRef(id="arxiv:1706.03762", title="Attention Is All You Need", year=2017,
                     url="https://arxiv.org/abs/1706.03762", source="arxiv")
    chats.append_message(a.chat_id, ChatMessage(role="user", content="find papers on attention"))
    chats.append_message(a.chat_id, ChatMessage(role="agent", content="Here are the papers I found.", papers=[paper]))

    turns_a = chats.history_turns(chats.get_chat(a.chat_id), 10)
    assert "1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017)" in turns_a[0].assistant
    assert chats.history_turns(chats.get_chat(b.chat_id), 10) == []
