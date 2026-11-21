"""
app.py
------
ScholarPulse agent UI: named chats (saved by the backend in MongoDB, or locally
when it is offline), global PDF uploads, and answers with citations,
faithfulness, papers found, plots and the agent's tool steps.

Run from research_assistant/:
    streamlit run frontend/app.py --server.port 8501

The original report generator is still available via frontend/streamlit_app.py.
"""
from __future__ import annotations

import base64
import html
import json
from functools import lru_cache
from pathlib import Path

import requests
import streamlit as st

from utils.api import (
    API_BASE,
    agent_chat,
    create_chat,
    delete_chat,
    delete_document,
    error_detail,
    get_chat,
    ingest_paper,
    list_chats,
    list_documents,
    rename_chat,
    storage_status,
    upload_document,
)
from utils.session import (
    active_chat_id,
    init_agent_session,
    is_expanded,
    mark_chat_not_found,
    set_active_chat,
    take_chat_not_found,
    toggle_expanded,
)
from utils.styles import load_css, theme_toggle

GOOD, MID = 0.85, 0.7  # badge colour thresholds
ICON_PATH = Path(__file__).resolve().parent / "images" / "heartbeat.png"  # transparent PNG: works in light + dark
NAME_MAX_CHARS = 60
LONG_MESSAGE_CHARS = 300  # user messages longer than this start collapsed
PREVIEW_LINES = 3


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


# ── Chat dialogs ─────────────────────────────────────────────────────────────


@st.dialog("New chat")
def _new_chat_dialog() -> None:
    name = st.text_input("Name", max_chars=NAME_MAX_CHARS, placeholder="e.g. Dehazing survey")
    if st.button("Create", type="primary"):
        if not name.strip():
            st.error("A name is required.")
            return
        try:
            with st.spinner("Creating chat..."):
                chat = create_chat(name.strip())
        except requests.RequestException as exc:
            st.error(error_detail(exc))
            return
        set_active_chat(chat["chat_id"])
        st.rerun()


@st.dialog("Rename chat")
def _rename_dialog(chat: dict) -> None:
    name = st.text_input("Name", value=chat["name"], max_chars=NAME_MAX_CHARS)
    if st.button("Save", type="primary"):
        if not name.strip():
            st.error("A name is required.")
            return
        try:
            with st.spinner("Saving..."):
                rename_chat(chat["chat_id"], name.strip())
        except requests.RequestException as exc:
            st.error(error_detail(exc))
            return
        st.rerun()


@st.dialog("Delete chat?")
def _delete_dialog(chat: dict) -> None:
    try:
        with st.spinner("Loading..."):
            n_docs = len(list_documents(chat["chat_id"]))
    except requests.RequestException:
        n_docs = 0
    docs = f" and its {n_docs} document{'s' if n_docs != 1 else ''}" if n_docs else ""
    st.write(f"Delete **{chat['name']}**, all its messages{docs}? This can't be undone.")
    confirm, cancel = st.columns(2)
    if confirm.button("Delete", type="primary", width="stretch"):
        try:
            with st.spinner("Deleting chat and its documents..."):
                delete_chat(chat["chat_id"])
        except requests.RequestException as exc:
            st.error(error_detail(exc))
            return
        if active_chat_id() == chat["chat_id"]:
            set_active_chat(None)
        st.rerun()
    if cancel.button("Cancel", width="stretch"):
        st.rerun()


@st.dialog("Remove PDF?")
def _remove_doc_dialog(doc: dict, chat_id: str) -> None:
    st.write(f"Remove **{doc['title']}**?")
    st.caption("Earlier answers keep their citations.")
    confirm, cancel = st.columns(2)
    if confirm.button("Remove", type="primary", width="stretch"):
        try:
            with st.spinner("Removing..."):
                delete_document(doc["doc_id"], chat_id)
        except requests.RequestException as exc:
            st.error(error_detail(exc))
            return
        st.rerun()
    if cancel.button("Cancel", width="stretch"):
        st.rerun()


# ── Sidebar ──────────────────────────────────────────────────────────────────


def _render_chat_list() -> None:
    try:
        with st.spinner("Loading chats..."):
            chats = list_chats()
    except requests.RequestException:
        st.warning("Backend unreachable at the API base URL.")
        return
    if not chats:
        st.caption("No chats yet.")
    for chat in chats:
        cid = chat["chat_id"]
        is_active = cid == active_chat_id()
        # Keys drive the CSS: st-key-chatrow-* (row), st-key-open[act]-* (name), st-key-chatmenu-* (⋮).
        with st.container(key=f"chatrow-{cid}"):
            name_col, menu_col = st.columns([6, 1], gap="small", vertical_alignment="center")
            if name_col.button(chat["name"], key=f"{'openact' if is_active else 'open'}-{cid}",
                               width="stretch", help=chat["name"]):
                set_active_chat(cid)
                st.rerun()
            with menu_col.popover("⋮", key=f"chatmenu-{cid}", help="Rename or delete"):
                if st.button("Rename", key=f"rename-{cid}", width="stretch"):
                    _rename_dialog(chat)
                if st.button("Delete", key=f"delete-{cid}", width="stretch"):
                    _delete_dialog(chat)


def _render_documents() -> None:
    """The open chat's PDFs: upload box plus the list, each card with a remove button."""
    chat_id = active_chat_id()
    if chat_id is None:
        st.caption("Create or open a chat to add PDFs")
        return
    files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True, key=f"uploader-{chat_id}")
    for f in files or []:
        sent_key = f"{chat_id}:{f.name}"
        if sent_key in st.session_state["agent_uploaded"]:
            continue
        with st.spinner(f"Ingesting {f.name}…"):
            try:
                info = upload_document(f.name, f.getvalue(), chat_id)
            except requests.RequestException as exc:
                st.error(f"{f.name}: {error_detail(exc)}")
                continue
        if info.get("skipped"):
            st.info(f"{f.name} is already in this chat.")
        else:
            st.success(f"{f.name}: {info['chunk_count']} chunks indexed")
        st.session_state["agent_uploaded"].append(sent_key)

    try:
        with st.spinner("Loading documents..."):
            docs = list_documents(chat_id)
    except requests.RequestException:
        return
    if not docs:
        st.caption("No PDFs in this chat yet.")
    for d in docs:
        # A card (st-key-doccard-*) with the × button pinned to its top-right corner (st-key-docx-*).
        with st.container(key=f"doccard-{d['doc_id']}"):
            st.markdown(f"<div class='sp-doc-name' title='{html.escape(d['title'])}'>{html.escape(d['title'])}</div>"
                        f"<div class='sp-doc-meta'>{d['chunk_count']} chunks</div>", unsafe_allow_html=True)
            if st.button("×", key=f"docx-{d['doc_id']}", help="Remove from this chat", type="tertiary"):
                _remove_doc_dialog(d, chat_id)


def render_sidebar() -> None:
    with st.sidebar:
        try:
            if not storage_status()["online"]:
                st.markdown("<div class='sp-offline'>offline: saving locally</div>", unsafe_allow_html=True)
        except requests.RequestException:
            pass
        if st.button("New chat", icon=":material/add:", width="stretch", type="primary"):
            _new_chat_dialog()
        st.markdown("<div class='sp-section-label'>Chats</div>", unsafe_allow_html=True)
        _render_chat_list()
        st.divider()
        st.subheader("Documents")
        _render_documents()


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
                    out = ingest_paper(p, active_chat_id())
                except requests.RequestException as exc:
                    st.error(f"{p['title']}: {error_detail(exc)}")
                    continue
            if out["status"] == "failed":
                st.error(f"{p['title']}: {out.get('detail') or 'ingest failed'}")
            elif out["status"] == "already_ingested":
                st.info(f"{p['title']} is already indexed.")
            else:
                st.toast(f"{p['title']}: {out['chunk_count']} chunks indexed")
                st.rerun()  # refresh the sidebar's document list


NOTE_LABELS = {"reask": "Re-ask", "final_thought": "Final thought", "step_limit": "Step limit", "routing": "Routing"}


def _render_step(i: int, step: dict) -> None:
    meta = [m for m in (step.get("model_used"), f"{step['total_ms'] / 1000:.1f}s" if step.get("total_ms") else None) if m]
    st.markdown(f"**Step {i}: {step.get('tool') or 'no tool'}**" + (f" · {' · '.join(meta)}" if meta else ""))
    if step.get("thought"):
        st.caption("Thought")
        st.code(step["thought"], language=None, wrap_lines=True)
    if step.get("args"):
        st.caption("Args")
        st.code(json.dumps(step["args"], indent=2, ensure_ascii=False), language="json", wrap_lines=True)
    st.caption("Observation")
    st.code(step.get("observation", ""), language=None, wrap_lines=True)


def _render_agent_steps(result: dict) -> None:
    """Collapsed "Agent steps (N)": every tool step, then answer model/latency and agent notes."""
    steps, notes = result.get("steps") or [], result.get("notes") or []
    with st.expander(f"Agent steps ({len(steps)})", expanded=False):
        if not steps:
            st.caption("No tools were called for this answer.")
        for i, step in enumerate(steps, 1):
            _render_step(i, step)
        answer_meta = [m for m in (result.get("model_used"),
                                   f"{result['latency_ms'] / 1000:.1f}s" if result.get("latency_ms") is not None else None) if m]
        if answer_meta:
            st.markdown(f"**Final answer** · {' · '.join(answer_meta)}")
        other = [n for n in notes if n.get("kind") != "routing"]
        if other:
            st.caption("Notes")
            for n in other:
                st.code(f"[{NOTE_LABELS.get(n['kind'], n['kind'])}, before step {n['step']}] {n['text']}",
                        language=None, wrap_lines=True)


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

    _render_agent_steps(result)


# ── Messages ─────────────────────────────────────────────────────────────────


def _bubble_html(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def _preview(text: str) -> str:
    """First 3 lines (and at most 300 characters) of a long message."""
    head = "\n".join(text.splitlines()[:PREVIEW_LINES])
    return (head[:LONG_MESSAGE_CHARS].rstrip() if len(head) > LONG_MESSAGE_CHARS else head) + " …"


def render_user_message(content: str, key: str) -> None:
    """Right-aligned bubble; long messages start collapsed with a Show more / Show less toggle."""
    is_long = len(content) > LONG_MESSAGE_CHARS
    full = not is_long or is_expanded(key)
    shown = content if full else _preview(content)
    st.markdown(f"<div class='sp-user-row'><div class='sp-user-bubble'>{_bubble_html(shown)}</div></div>",
                unsafe_allow_html=True)
    if is_long:
        _, toggle = st.columns([5, 1])
        toggle.button("Show less" if full else "Show more", key=f"more-{key}",
                      on_click=toggle_expanded, args=(key,), type="tertiary")


def render_message(msg: dict, key: str) -> None:
    if msg["role"] == "user":
        render_user_message(msg["content"], key)
    else:
        with st.container(key=f"agentmsg-{key}"):  # plain, left-aligned
            render_answer({**msg, "answer": msg["content"]}, key)


@lru_cache(maxsize=1)
def _icon_data_uri() -> str:
    """The project icon inlined, so the navbar HTML can show it without a static file server."""
    return "data:image/png;base64," + base64.b64encode(ICON_PATH.read_bytes()).decode()


def render_navbar() -> None:
    """Sticky top bar on every page: project name on the left, theme toggle on the right."""
    with st.container(key="navbar"):
        brand, actions = st.columns([5, 1], gap="small", vertical_alignment="center")
        brand.markdown(
            f"<div class='sp-brand'><img class='sp-brand-mark' src='{_icon_data_uri()}' alt='ScholarPulse logo'>"
            "<span class='sp-brand-name'>ScholarPulse</span>"
            "<span class='sp-brand-tag'>Research assistant</span></div>",
            unsafe_allow_html=True,
        )
        with actions:
            theme_toggle()


def main() -> None:
    st.set_page_config(page_title="ScholarPulse", page_icon=str(ICON_PATH), layout="wide")
    init_agent_session()
    load_css()
    render_navbar()

    # Resolve ?chat= before the sidebar, so a missing chat isn't shown as open there.
    chat_id = active_chat_id()
    chat = None
    if chat_id is not None:
        try:
            with st.spinner("Loading chat..."):
                chat = get_chat(chat_id)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in (404, 422):
                mark_chat_not_found()  # deleted or wrong id: clears ?chat=
                chat_id = None
            else:
                st.error(error_detail(exc))
        except requests.RequestException as exc:
            st.error(error_detail(exc))
    render_sidebar()

    if chat is None:
        if take_chat_not_found():
            st.warning("Chat not found")
        st.markdown("<div class='sp-empty'><div class='sp-empty-title'>Start a research chat</div>"
                    "<div class='sp-empty-text'>Create one with <b>New chat</b> in the sidebar, or open an "
                    "existing chat. Each chat keeps its own PDFs.</div></div>", unsafe_allow_html=True)
        st.chat_input("Open a chat to start asking…", disabled=True)
        return

    st.subheader(chat["name"])
    for i, msg in enumerate(chat["messages"]):
        render_message(msg, key=f"{chat_id}-{i}")

    prompt = st.chat_input("Ask a question about your papers…")
    if not prompt:
        return
    render_user_message(prompt, key=f"{chat_id}-{len(chat['messages'])}")
    with st.container():
        with st.spinner("Researching...", show_time=True):
            try:
                agent_chat(chat_id, prompt)
            except requests.RequestException as exc:
                st.error(error_detail(exc))
                return
    st.rerun()  # re-render from the stored chat (both messages are saved by the backend)


main()
