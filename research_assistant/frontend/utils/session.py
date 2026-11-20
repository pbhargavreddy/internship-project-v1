"""
utils/session.py
----------------
Session state initialisation plus helpers for job, chat-history and project
state shared across components.
"""
from __future__ import annotations

import uuid

import streamlit as st

# ── Default values ────────────────────────────────────────────────────────────

_JOB_DEFAULTS: dict[str, object] = {
    "job_id": None,
    "job_status": None,
    "job_result": None,
    "job_error": None,
    "job_topic": "",
    "job_started_at": None,
}

_SIDEBAR_DEFAULTS: dict[str, object] = {
    "chat_histories": [],   # list of {"id", "title", "result", "job_id"}
    "projects": [],         # list of {"id", "name"}
    "active_chat": None,
    "active_project": None,
}


def init_session() -> None:
    """Idempotently initialise all session state keys."""
    for key, val in {**_JOB_DEFAULTS, **_SIDEBAR_DEFAULTS}.items():
        st.session_state.setdefault(key, val)


def reset_job() -> None:
    """Clear job-related session state so a new query can start."""
    for key, val in _JOB_DEFAULTS.items():
        st.session_state[key] = val


# ── Chat history ──────────────────────────────────────────────────────────────

def add_chat(title: str, result: str = "", job_id: str | None = None) -> str:
    """Add a chat to the top of history, mark it active and return its id."""
    cid = str(uuid.uuid4())
    st.session_state["chat_histories"].insert(
        0, {"id": cid, "title": title, "result": result, "job_id": job_id}
    )
    st.session_state["active_chat"] = cid
    return cid


def update_chat(chat_id: str | None, **fields: object) -> None:
    """Update fields (e.g. result) on an existing chat entry, if present."""
    if chat_id is None:
        return
    for chat in st.session_state["chat_histories"]:
        if chat["id"] == chat_id:
            chat.update(fields)
            return


def open_chat(chat: dict) -> None:
    """Load a stored chat back into the job state so its output is shown."""
    result = chat.get("result") or None
    st.session_state["active_chat"] = chat["id"]
    st.session_state["job_id"] = chat.get("job_id")
    st.session_state["job_topic"] = chat.get("title", "")
    st.session_state["job_result"] = result
    st.session_state["job_error"] = None
    st.session_state["job_status"] = "completed" if result else None


# ── Projects ──────────────────────────────────────────────────────────────────

def add_project(name: str) -> None:
    """Add a new project to the top of the list."""
    st.session_state["projects"].insert(0, {"id": str(uuid.uuid4()), "name": name})


# ── Agent chat (app.py) ───────────────────────────────────────────────────────

_AGENT_DEFAULTS: dict[str, object] = {
    "agent_messages": [],   # list of {"role", "content", "result"}
    "agent_uploaded": [],   # names of files already sent to /documents/upload
}


def init_agent_session() -> None:
    """Initialise agent-chat keys; the session_id is created once per browser session."""
    for key, val in _AGENT_DEFAULTS.items():
        st.session_state.setdefault(key, list(val) if isinstance(val, list) else val)
    st.session_state.setdefault("agent_session_id", uuid.uuid4().hex[:16])


def add_agent_message(role: str, content: str, result: dict | None = None) -> None:
    """Append a chat message (``result`` is the full /agent/chat response for answers)."""
    st.session_state["agent_messages"].append({"role": role, "content": content, "result": result})


def new_agent_session() -> None:
    """Start a fresh conversation (new backend memory)."""
    st.session_state["agent_messages"] = []
    st.session_state["agent_session_id"] = uuid.uuid4().hex[:16]
