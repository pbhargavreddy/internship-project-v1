"""
streamlit_app.py
----------------
Entry point for the AI Research Assistant Streamlit frontend.

Structure
---------
  streamlit_app.py          ← routing & layout entry point
  app_pages/
    sidebar.py              ← sidebar: chat history, projects, account
    input_panel.py          ← centered search bar + modal preview + expanding output
    results.py              ← polling status for background jobs
  utils/
    api.py                  ← FastAPI HTTP helpers
    demo_generator.py       ← demo report generation from demo_research_output.txt
    file_parser.py          ← document text extraction + FileInfo + preview check
    session.py              ← session state, chat-history and project helpers
    styles.py               ← loads styles/main.css + CSS-class helpers
  styles/
    main.css                ← all custom CSS (no inline styles in components)
  .streamlit/
    config.toml             ← dual light/dark theme (follows system by default)

Run
---
  streamlit run research_assistant/frontend/streamlit_app.py --server.port 8501
"""

import streamlit as st

from utils.session import init_session
from utils.styles import load_css
from app_pages.sidebar import render_sidebar
from app_pages.input_panel import render_input_panel
from app_pages.results import render_results

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

init_session()
load_css()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

render_sidebar()

# ─────────────────────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────────────────────

# Always render the centered input panel. When output is produced, the div
# expands downwards directly below the search bar.
render_input_panel()

# If an asynchronous background job is still running/polling from FastAPI backend,
# render the status poller below
if st.session_state.get("job_status") in ("pending", "running"):
    render_results()
