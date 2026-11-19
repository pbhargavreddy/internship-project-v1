"""
app_pages/results.py
--------------------
Polling handler for background jobs running on the FastAPI backend.
When a job finishes, it triggers a rerun so the expanding output div in
input_panel displays the full report.
"""
from __future__ import annotations

import time
from datetime import datetime

import requests
import streamlit as st

from utils.api import poll_job, POLL_INTERVAL_S, TERMINAL_STATUSES
from utils.session import update_chat


def render_results() -> None:
    """Render the active background polling status widget."""
    job_id: str | None = st.session_state.get("job_id")
    if job_id is None:
        return

    # ── Active polling ─────────────────────────────────────────────────────────
    if st.session_state.get("job_status") not in TERMINAL_STATUSES:
        _, center, _ = st.columns([1, 4, 1])
        with center:
            with st.status("Research in progress, synthesizing knowledge...", expanded=True) as status_widget:
                st.write(f"**Topic:** {st.session_state.get('job_topic')}")
                st.write(f"**Job ID:** `{job_id}`")

                while True:
                    try:
                        data = poll_job(job_id)
                    except requests.exceptions.RequestException as exc:
                        st.session_state["job_status"] = "failed"
                        st.session_state["job_error"] = f"Polling error: {exc}"
                        break

                    current = data["status"]
                    st.session_state["job_status"] = current
                    st.write(
                        f"Status: **{current}** — "
                        f"checked at {datetime.now().strftime('%H:%M:%S')}"
                    )

                    if current in TERMINAL_STATUSES:
                        st.session_state["job_result"] = data.get("result")
                        st.session_state["job_error"] = data.get("error")
                        # Persist the finished result into the active chat entry
                        update_chat(
                            st.session_state.get("active_chat"),
                            result=data.get("result") or "",
                        )
                        break

                    time.sleep(POLL_INTERVAL_S)

                if st.session_state["job_status"] == "completed":
                    status_widget.update(
                        label="Research complete.",
                        state="complete",
                        expanded=False,
                    )
                    st.rerun()
                else:
                    status_widget.update(
                        label="Research job failed.",
                        state="error",
                        expanded=False,
                    )

    # If failed, show error in center column
    elif st.session_state.get("job_status") == "failed":
        _, center, _ = st.columns([1, 4, 1])
        with center:
            st.error(
                st.session_state.get("job_error", "Unknown error occurred"),
                title="Research job failed",
            )
