"""
app_pages/input_panel.py
------------------------
Centered Gemini-style input panel with:
- Lower vertical placement when idle, expanding downwards when output is ready.
- Rotating placeholder prompts.
- File uploader without inline preview: uploaded files appear as clickable chips.
- Modal preview in the center when clicking a file; if preview is not possible,
  shows 'No preview available' at the search bar in red text for 2 seconds.
- Enter key or the arrow button submits the query.
- Demo report generation from demo_research_output.txt when no backend is up.
"""
from __future__ import annotations

import html
import random
import time
from datetime import datetime

import requests
import streamlit as st

from utils.api import post_research
from utils.demo_generator import get_demo_report
from utils.file_parser import (
    ACCEPTED_EXTENSIONS,
    FileInfo,
    fmt_size,
    is_preview_available,
    open_image,
    parse_uploaded_file,
)
from utils.session import add_chat, reset_job
from utils.styles import spacer

# ── Rotating placeholder text (Gemini-style) ─────────────────────────────────

_PLACEHOLDERS = [
    "What's on your mind?",
    "What shall we dig into today?",
    "Ask me anything to research...",
    "Drop a topic and let's explore.",
    "What do you want to know more about?",
    "Start a research thread...",
]


def _placeholder() -> str:
    """Return a consistent placeholder per query (not re-randomised on rerun)."""
    if "input_placeholder" not in st.session_state:
        st.session_state["input_placeholder"] = random.choice(_PLACEHOLDERS)
    return st.session_state["input_placeholder"]


# ── Center modal dialog for file preview ─────────────────────────────────────

@st.dialog("Document Preview")
def _show_preview_dialog(fi: FileInfo) -> None:
    """Centered modal dialog showing the document preview."""
    st.markdown(f"**{html.escape(fi.name)}**")
    st.caption(f"{fi.ext.upper()} · {fmt_size(fi.size_bytes)}")
    st.divider()

    if fi.is_image and fi.image_bytes:
        st.image(open_image(fi.image_bytes), width="stretch")
    elif fi.text_excerpt:
        # Escape file content: it is untrusted and rendered as HTML.
        st.markdown(
            f"<div class='ra-preview-text'>{html.escape(fi.text_excerpt)}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("No preview content available.")


def _flash_no_preview(placeholder, name: str) -> None:
    """Show a red message that hides itself after 2s via CSS (non-blocking).

    A per-click counter makes the markup unique so the animation restarts
    even when the same file is clicked twice in a row.
    """
    n = st.session_state.get("_flash_n", 0) + 1
    st.session_state["_flash_n"] = n
    placeholder.markdown(
        f"<div class='ra-flash' data-n='{n}'>No preview available for {html.escape(name)}</div>",
        unsafe_allow_html=True,
    )


# ── Main component ────────────────────────────────────────────────────────────

def render_input_panel() -> None:
    """Render the centered Gemini-style input panel and expanding output div."""
    has_output = bool(st.session_state.get("job_result"))
    spacer("ra-spacer-active" if has_output else "ra-spacer-idle")

    _, center, _ = st.columns([1, 4, 1])
    with center:
        if not has_output:
            st.markdown("### Research Assistant")
            st.caption("Ask questions, explore academic topics, or inspect uploaded literature.")
            st.write("")

        with st.container(border=True):
            _render_attachments()
            topic_input, submitted = _render_query_form()

        if submitted:
            _handle_submit(topic_input)

        if has_output:
            _render_expanding_output()


def _render_attachments() -> None:
    """File uploader plus clickable chips for each attached file."""
    uploaded_files = st.file_uploader(
        label="Attach files",
        type=ACCEPTED_EXTENSIONS,
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="uploader",
    )
    flash = st.empty()

    if not uploaded_files:
        return

    st.markdown("<p class='ra-hint'>Attached files (click to preview):</p>", unsafe_allow_html=True)
    with st.container(horizontal=True):
        for idx, uf in enumerate(uploaded_files):
            fi = parse_uploaded_file(uf)
            icon = ":material/image:" if fi.is_image else ":material/description:"
            if st.button(
                f"{icon} {fi.name} ({fmt_size(fi.size_bytes)})",
                key=f"chip_{idx}_{uf.name}",
                type="secondary",
            ):
                if is_preview_available(fi):
                    _show_preview_dialog(fi)
                else:
                    _flash_no_preview(flash, fi.name)


def _render_query_form() -> tuple[str, bool]:
    """Text input + return button inside a form so Enter also submits."""
    with st.form("query_form", border=False, clear_on_submit=True, enter_to_submit=True):
        input_col, btn_col = st.columns([12, 1], vertical_alignment="bottom")
        with input_col:
            topic_input = st.text_input(
                label="Research topic",
                placeholder=_placeholder(),
                label_visibility="collapsed",
                key="topic_input",
            )
        with btn_col:
            submitted = st.form_submit_button(
                ":material/arrow_upward:",
                type="primary",
                help="Start research",
            )
    return topic_input, submitted


def _handle_submit(topic_input: str) -> None:
    """Submit to the FastAPI backend, falling back to the demo report."""
    query = topic_input.strip()
    if not query:
        st.warning("Please enter a research topic.")
        return

    reset_job()
    st.session_state.pop("input_placeholder", None)
    st.session_state["job_topic"] = query
    st.session_state["job_started_at"] = datetime.now()

    result_report = ""
    try:
        data = post_research(query)
        st.session_state["job_id"] = data["job_id"]
        st.session_state["job_status"] = data["status"]
    except (requests.exceptions.RequestException, KeyError, ValueError):
        # Backend not running or returned something unexpected — use demo text.
        with st.spinner("Synthesizing research findings..."):
            time.sleep(0.6)  # subtle transition
            result_report = get_demo_report(query)
        st.session_state["job_id"] = f"demo-{int(time.time())}"
        st.session_state["job_status"] = "completed"
        st.session_state["job_result"] = result_report

    # Backend jobs get their result filled in by results.py when they finish.
    add_chat(query, result_report, job_id=st.session_state["job_id"])
    st.rerun()


def _render_expanding_output() -> None:
    """Render the research output expanding downwards below the search bar."""
    result_md: str = st.session_state.get("job_result", "")
    topic: str = st.session_state.get("job_topic", "")
    job_id: str = st.session_state.get("job_id") or "—"

    st.write("")
    with st.container(border=True):
        col_title, col_action = st.columns([5, 1], vertical_alignment="center")
        with col_title:
            st.markdown(f"#### {topic}")
            st.caption(f"Session `{job_id}` · Completed")
        with col_action:
            st.download_button(
                label=":material/download: Export .md",
                data=result_md.encode("utf-8"),
                file_name=f"research_{topic[:30].replace(' ', '_')}.md",
                mime="text/markdown",
                type="secondary",
                key="btn_download_report",
                width="stretch",
            )
        st.divider()
        st.markdown(result_md)
