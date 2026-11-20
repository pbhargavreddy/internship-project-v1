"""
app.py
------
ScholarPulse agent UI: upload PDFs, chat with the ReAct agent, and inspect
citations, faithfulness and the agent's tool steps.

Run from research_assistant/:
    streamlit run frontend/app.py --server.port 8501

The original report generator is still available via frontend/streamlit_app.py.
"""
from __future__ import annotations

import html
from pathlib import Path

import requests
import streamlit as st

from utils.api import API_BASE, agent_chat, error_detail, ingest_paper, list_documents, upload_document
from utils.session import add_agent_message, init_agent_session, new_agent_session
from utils.styles import load_css

GOOD, MID = 0.85, 0.7  # badge colour thresholds


def _badge(faith: dict | None) -> str:
    if not faith:
        return "<span class='sp-badge sp-badge-none'>Faithfulness: n/a</span>"
    score = faith["score"]
    level = "good" if score >= GOOD else "mid" if score >= MID else "bad"
    return f"<span class='sp-badge sp-badge-{level}'>Faithfulness: {score:.0%}</span>"


def _model_badge(result: dict) -> str:
    """Which model wrote the answer: fine-tuned (with latency), Groq, or Groq after a fallback."""
    model = result.get("model_used") or ""
    if not model:
        return ""
    if result.get("fallback_reason"):
        provider = model.split()[0]
        text, kind = f"{provider} · fallback: {result['fallback_reason']}", "fallback"
    elif result.get("latency_ms") is not None:
        text, kind = f"{model} · {result['latency_ms'] / 1000:.1f}s", "model"
    else:
        text, kind = model, "model"
    return f" <span class='sp-badge sp-badge-{kind}'>{html.escape(text)}</span>"


def render_sidebar() -> None:
    with st.sidebar:
        st.title("ScholarPulse")
        if st.button("New conversation", use_container_width=True):
            new_agent_session()
            st.rerun()
        st.caption(f"Session: `{st.session_state['agent_session_id']}`")

        st.subheader("Documents")
        files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        for f in files or []:
            if f.name in st.session_state["agent_uploaded"]:
                continue
            with st.spinner(f"Ingesting {f.name}…"):
                try:
                    info = upload_document(f.name, f.getvalue())
                    st.success(f"{f.name}: {info['chunk_count']} chunks indexed")
                except requests.RequestException as exc:
                    st.error(f"{f.name}: {error_detail(exc)}")
                    continue
            st.session_state["agent_uploaded"].append(f.name)

        try:
            docs = list_documents()
        except requests.RequestException:
            st.warning("Backend unreachable at the API base URL.")
            return
        if not docs:
            st.caption("No documents yet.")
        for d in docs:
            st.markdown(f"📄 **{d['title']}**  \n<small>{d['chunk_count']} chunks</small>", unsafe_allow_html=True)


def _render_images(paths: list[str]) -> None:
    """Show plots: from disk when the backend is local, else from the backend's /plots route."""
    for path in paths:
        st.image(path if Path(path).is_file() else f"{API_BASE}/plots/{Path(path).name}")


def _md_link(text: str, url: str | None) -> str:
    safe = text.replace("[", "\\[").replace("]", "\\]")
    return f"[{safe}]({url})" if url else safe


def _render_papers(papers: list[dict], key: str) -> None:
    """'Papers found' table: linked title, year, authors, PDF link, citations, Ingest button."""
    st.markdown("**Papers found**")
    widths = [6, 1, 3, 1, 1, 1.4]
    for col, label in zip(st.columns(widths), ["Title", "Year", "Authors", "PDF", "Cited", ""]):
        col.caption(label)
    for i, p in enumerate(papers):
        title, year, authors, pdf, cited, action = st.columns(widths)
        title.markdown(_md_link(p["title"], p.get("url")))
        year.write(p.get("year") or "n.d.")
        names = p.get("authors") or []
        authors.write(", ".join(names[:3]) + (" et al." if len(names) > 3 else "") or "—")
        pdf.markdown(_md_link("PDF", p["pdf_url"]) if p.get("pdf_url") else "—")
        cited.write(p.get("citation_count") if p.get("citation_count") is not None else "—")
        if action.button("Ingest", key=f"ingest-{key}-{i}", disabled=not p.get("pdf_url"),
                         help="Download and index this PDF so rag_search can read it"):
            with st.spinner(f"Ingesting {p['title']}…"):
                try:
                    out = ingest_paper(p)
                except requests.RequestException as exc:
                    st.error(f"{p['title']}: {error_detail(exc)}")
                    continue
            if out["status"] == "failed":
                st.error(f"{p['title']}: {out.get('detail') or 'ingest failed'}")
            elif out["status"] == "already_ingested":
                st.info(f"{p['title']} is already indexed.")
            else:
                st.success(f"{p['title']}: {out['chunk_count']} chunks indexed")


def render_answer(result: dict, key: str = "new") -> None:
    st.markdown(result["answer"])
    _render_images(result.get("images") or [p for step in result.get("steps", []) for p in step.get("images", [])])
    badges = _badge(result.get("faithfulness")) + _model_badge(result)
    st.markdown(badges, unsafe_allow_html=True)

    faith = result.get("faithfulness") or {}
    if faith.get("unsupported"):
        with st.expander(f"Unsupported sentences ({len(faith['unsupported'])})"):
            for s in faith["unsupported"]:
                st.markdown(f"- {s}")

    if result.get("citations"):
        items = "".join(
            f"<li>[Doc {c['doc']}] {html.escape(c['title'])}"
            + (f", p.{c['page']}" if c.get("page") else "")
            + (f", {html.escape(c['section'])}" if c.get("section") else "")
            + "</li>"
            for c in result["citations"]
        )
        st.markdown(f"<div class='sp-citations'><b>Citations</b><ul>{items}</ul></div>", unsafe_allow_html=True)

    if result.get("papers"):
        _render_papers(result["papers"], key)

    if result.get("steps"):
        with st.expander(f"Agent steps ({len(result['steps'])})"):
            for i, step in enumerate(result["steps"], 1):
                model = f" · {step['model_used']}" if step.get("model_used") else ""
                st.markdown(f"**Step {i}: {step.get('tool') or 'no tool'}{model}**")
                if step.get("thought"):
                    st.markdown(f"*Thought:* {step['thought']}")
                if step.get("args"):
                    st.json(step["args"], expanded=False)
                st.code(step.get("observation", ""), language=None, wrap_lines=True)


def main() -> None:
    st.set_page_config(page_title="ScholarPulse Agent", page_icon="🔬", layout="wide")
    init_agent_session()
    load_css()
    render_sidebar()

    st.header("Ask about your papers")
    for i, msg in enumerate(st.session_state["agent_messages"]):
        with st.chat_message(msg["role"]):
            if msg["result"]:
                render_answer(msg["result"], key=str(i))
            else:
                st.markdown(msg["content"])

    prompt = st.chat_input("Ask a question about the uploaded papers…")
    if not prompt:
        return
    add_agent_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Researching…"):
            try:
                result = agent_chat(st.session_state["agent_session_id"], prompt)
            except requests.RequestException as exc:
                msg = f"⚠️ {error_detail(exc)}"
                st.error(msg)
                add_agent_message("assistant", msg)
                return
        # Same key it gets when re-rendered from history (its index once appended).
        render_answer(result, key=str(len(st.session_state["agent_messages"])))
    add_agent_message("assistant", result["answer"], result)


main()
