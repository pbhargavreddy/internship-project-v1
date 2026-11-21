"""
utils/api.py
------------
API helpers for communicating with the FastAPI research backend.
"""
from __future__ import annotations

import os

import requests

API_BASE = os.environ.get("SCHOLARPULSE_API_BASE", "http://localhost:8000")
AGENT_TIMEOUT_S = 300  # agent turns run several LLM + tool calls
INGEST_TIMEOUT_S = 180  # PDF download + chunking + embedding
POLL_INTERVAL_S = 2
TERMINAL_STATUSES = frozenset({"completed", "failed"})


def post_research(topic: str) -> dict:
    """Submit a new research job. Returns {job_id, status}."""
    r = requests.post(f"{API_BASE}/research", json={"topic": topic}, timeout=10)
    r.raise_for_status()
    return r.json()


def poll_job(job_id: str) -> dict:
    """Fetch current status of a job. Returns {job_id, status, topic, result, error}."""
    r = requests.get(f"{API_BASE}/research/{job_id}", timeout=10)
    r.raise_for_status()
    return r.json()



def upload_document(name: str, data: bytes, chat_id: str) -> dict:
    """Upload and ingest a PDF into a chat. Returns {doc_id, filename, chunk_count, skipped}."""
    r = requests.post(
        f"{API_BASE}/documents/upload", params={"chat_id": chat_id},
        files={"file": (name, data, "application/pdf")}, timeout=AGENT_TIMEOUT_S,
    )
    r.raise_for_status()
    return r.json()


def list_documents(chat_id: str) -> list[dict]:
    """A chat's documents: [{doc_id, title, chunk_count}]."""
    r = requests.get(f"{API_BASE}/documents", params={"chat_id": chat_id}, timeout=10)
    r.raise_for_status()
    return r.json()


def delete_document(doc_id: str, chat_id: str) -> dict:
    """Remove a document from a chat (the file goes once no chat uses it)."""
    r = requests.delete(f"{API_BASE}/documents/{doc_id}", params={"chat_id": chat_id}, timeout=30)
    r.raise_for_status()
    return r.json()


def agent_chat(chat_id: str, message: str) -> dict:
    """One agent turn in a stored chat (the backend saves both messages).

    Returns:
        ``{answer, citations, steps, papers, images, model_used, faithfulness, ...}``.

    Raises:
        requests.HTTPError: e.g. 400 when the prompt-injection filter rejects the message.
    """
    r = requests.post(
        f"{API_BASE}/agent/chat", json={"chat_id": chat_id, "message": message}, timeout=AGENT_TIMEOUT_S
    )
    r.raise_for_status()
    return r.json()


def ingest_paper(paper: dict, chat_id: str) -> dict:
    """Download and index one paper from an agent response's ``papers`` list.

    Returns:
        ``{paper_id, title, status, doc_id, chunk_count, detail}``; status is
        ingested / already_ingested / failed.
    """
    r = requests.post(f"{API_BASE}/papers/ingest", params={"chat_id": chat_id}, json=paper, timeout=INGEST_TIMEOUT_S)
    r.raise_for_status()
    return r.json()


# ── Chats ────────────────────────────────────────────────────────────────────


def create_chat(name: str) -> dict:
    """Create a named chat. Returns ``{chat_id, name, created_at, updated_at}``."""
    r = requests.post(f"{API_BASE}/chats", json={"name": name}, timeout=10)
    r.raise_for_status()
    return r.json()


def list_chats() -> list[dict]:
    """Chats, most recently updated first (without messages)."""
    r = requests.get(f"{API_BASE}/chats", timeout=10)
    r.raise_for_status()
    return r.json()


def get_chat(chat_id: str) -> dict:
    """One chat with its ``messages``."""
    r = requests.get(f"{API_BASE}/chats/{chat_id}", timeout=10)
    r.raise_for_status()
    return r.json()


def rename_chat(chat_id: str, name: str) -> dict:
    r = requests.patch(f"{API_BASE}/chats/{chat_id}", json={"name": name}, timeout=10)
    r.raise_for_status()
    return r.json()


def delete_chat(chat_id: str) -> None:
    r = requests.delete(f"{API_BASE}/chats/{chat_id}", timeout=10)
    r.raise_for_status()


def storage_status() -> dict:
    """``{online, mode}``: whether chats go to MongoDB or are saved locally."""
    r = requests.get(f"{API_BASE}/health/storage", timeout=10)
    r.raise_for_status()
    return r.json()


def error_detail(exc: requests.RequestException) -> str:
    """Best human-readable message for a failed request (FastAPI ``detail`` if present)."""
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            return str(resp.json().get("detail", resp.text))
        except ValueError:
            return resp.text
    return str(exc)
