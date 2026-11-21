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
# Chats and their messages live in the backend (MongoDB, or local JSON offline);
# the browser session only remembers which chat is open and UI toggles.

_AGENT_DEFAULTS: dict[str, object] = {
    "agent_uploaded": [],      # names of files already sent to /documents/upload
    "expanded_messages": [],   # keys of long user messages currently shown in full
}


def init_agent_session() -> None:
    """Initialise agent-chat keys; the open chat is restored from the ``?chat=`` URL parameter."""
    for key, val in _AGENT_DEFAULTS.items():
        st.session_state.setdefault(key, list(val) if isinstance(val, list) else val)
    if "active_chat_id" not in st.session_state:
        from_url = st.query_params.get("chat")
        st.session_state["active_chat_id"] = from_url if from_url and from_url.isalnum() else None
        if from_url and not from_url.isalnum():  # malformed id: same as a missing chat
            mark_chat_not_found()


def mark_chat_not_found() -> None:
    """Close the requested chat, clear ``?chat=`` and show "Chat not found" on this run."""
    set_active_chat(None)
    st.session_state["chat_not_found"] = True


def take_chat_not_found() -> bool:
    """Whether "Chat not found" should be shown (reset once read)."""
    return bool(st.session_state.pop("chat_not_found", False))


def active_chat_id() -> str | None:
    """ID of the chat open in the main pane, if any."""
    return st.session_state.get("active_chat_id")


def set_active_chat(chat_id: str | None) -> None:
    """Open ``chat_id`` (``None`` closes the current chat); kept in the URL so a reload keeps it open."""
    st.session_state["active_chat_id"] = chat_id
    if chat_id:
        st.query_params["chat"] = chat_id
    else:
        st.query_params.pop("chat", None)


def is_expanded(key: str) -> bool:
    """Whether the long message ``key`` is shown in full."""
    return key in st.session_state["expanded_messages"]


def toggle_expanded(key: str) -> None:
    """Flip a long message between preview and full text."""
    keys = st.session_state["expanded_messages"]
    keys.remove(key) if key in keys else keys.append(key)
