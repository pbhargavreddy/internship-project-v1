"""
app_pages/sidebar.py
--------------------
Left sidebar component:
- Chat history list with "New chat" button.
- Projects list with "New project" button.
- Account info panel (shown via popover on click, bottom of sidebar).
"""
from __future__ import annotations

import streamlit as st

from utils.session import add_project, open_chat, reset_job
from utils.styles import spacer

_MAX_ITEMS = 10
_TITLE_MAX = 32


def _truncate(text: str, limit: int = _TITLE_MAX) -> str:
    return text[:limit] + "…" if len(text) > limit else text


def render_sidebar() -> None:
    """Render the full left sidebar."""
    with st.sidebar:
        st.markdown("### Research Assistant")
        st.divider()
        _render_chats()
        st.divider()
        _render_projects()
        spacer("ra-spacer-sidebar")
        st.divider()
        _render_account()


def _render_chats() -> None:
    if st.button(":material/add: New chat", key="btn_new_chat", width="stretch"):
        reset_job()
        st.session_state["active_chat"] = None
        st.rerun()

    st.caption("Recent chats")
    chats: list[dict] = st.session_state.get("chat_histories", [])
    if not chats:
        st.caption("No chats yet.")
        return

    active = st.session_state.get("active_chat")
    for chat in chats[:_MAX_ITEMS]:
        if st.button(
            _truncate(chat.get("title") or "Untitled chat"),
            key=f"chat_{chat['id']}",
            type="primary" if active == chat["id"] else "secondary",
            width="stretch",
        ):
            open_chat(chat)
            st.rerun()


def _render_projects() -> None:
    if st.button(":material/folder_open: New project", key="btn_new_project", width="stretch"):
        _new_project_dialog()

    st.caption("Projects")
    projects: list[dict] = st.session_state.get("projects", [])
    if not projects:
        st.caption("No projects yet.")
        return

    active = st.session_state.get("active_project")
    for proj in projects[:_MAX_ITEMS]:
        if st.button(
            f":material/folder: {_truncate(proj['name'])}",
            key=f"proj_{proj['id']}",
            type="primary" if active == proj["id"] else "secondary",
            width="stretch",
        ):
            st.session_state["active_project"] = proj["id"]
            st.rerun()


def _render_account() -> None:
    with st.popover(":material/account_circle: Account", width="stretch"):
        st.markdown("**Signed in as**")
        st.caption("user@example.com")
        st.divider()
        st.button(":material/settings: Settings", key="btn_settings", width="stretch")
        st.button(":material/logout: Sign out", key="btn_signout", width="stretch")


@st.dialog("New project")
def _new_project_dialog() -> None:
    """Modal dialog to name a new project."""
    name = st.text_input("Project name", placeholder="e.g. Transformer architectures")
    col1, col2 = st.columns(2)
    if col1.button("Create", type="primary", width="stretch"):
        if name.strip():
            add_project(name.strip())
            st.rerun()
        else:
            st.warning("Please enter a project name.")
    if col2.button("Cancel", width="stretch"):
        st.rerun()
