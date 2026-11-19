"""
utils/api.py
------------
API helpers for communicating with the FastAPI research backend.
"""
from __future__ import annotations

import requests

API_BASE = "http://localhost:8000"
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

