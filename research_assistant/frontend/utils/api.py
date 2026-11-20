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



def upload_document(name: str, data: bytes) -> dict:
    """Upload and ingest a PDF. Returns {doc_id, filename, chunk_count}."""
    r = requests.post(
        f"{API_BASE}/documents/upload", files={"file": (name, data, "application/pdf")}, timeout=AGENT_TIMEOUT_S
    )
    r.raise_for_status()
    return r.json()


def list_documents() -> list[dict]:
    """Uploaded documents: [{doc_id, title, chunk_count}]."""
    r = requests.get(f"{API_BASE}/documents", timeout=10)
    r.raise_for_status()
    return r.json()


def agent_chat(session_id: str, message: str) -> dict:
    """One agent turn. Returns {answer, citations, steps, faithfulness}.

    Raises:
        requests.HTTPError: e.g. 400 when the prompt-injection filter rejects the message.
    """
    r = requests.post(
        f"{API_BASE}/agent/chat", json={"session_id": session_id, "message": message}, timeout=AGENT_TIMEOUT_S
    )
    r.raise_for_status()
    return r.json()


def ingest_paper(paper: dict) -> dict:
    """Download and index one paper from an agent response's ``papers`` list.

    Returns:
        ``{paper_id, title, status, doc_id, chunk_count, detail}``; status is
        ingested / already_ingested / failed.
    """
    r = requests.post(f"{API_BASE}/papers/ingest", json=paper, timeout=INGEST_TIMEOUT_S)
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
