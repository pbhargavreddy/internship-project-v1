"""
main.py
-------
FastAPI application exposing two endpoints:

  POST /research          → Enqueue a background research job, return job_id
  GET  /research/{job_id} → Poll current status and retrieve result

Jobs are stored in an in-memory dict (suitable for a single-process server).
For multi-process / multi-worker deployments, swap the in-memory store for
Redis or a lightweight DB in a future sprint.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.llm_researcher import run_research_query

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------
# Schema per job:
#   {
#     "status": "pending" | "running" | "completed" | "failed",
#     "topic":  str,
#     "result": str | None,
#     "error":  str | None,
#   }

jobs: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _run_job(job_id: str, topic: str) -> None:
    """Execute the LLM research query and update the job store."""
    log.info("Job %s started — topic: %r", job_id, topic)
    jobs[job_id]["status"] = "running"
    try:
        result = run_research_query(topic)
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = result
        log.info("Job %s completed successfully", job_id)
    except Exception as exc:  # noqa: BLE001
        log.exception("Job %s failed: %s", job_id, exc)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(exc)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    log.info("Research Assistant API starting up")
    yield
    log.info("Research Assistant API shutting down")


app = FastAPI(
    title="AI Research Assistant API",
    description="Background LLM research job queue with status polling.",
    version="0.2.0",
    lifespan=lifespan,
)

# Allow the Streamlit frontend (any localhost origin) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=500, description="Research topic")


class JobCreatedResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    topic: str
    result: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/research",
    response_model=JobCreatedResponse,
    status_code=202,
    summary="Enqueue a background research job",
)
async def create_research_job(
    body: ResearchRequest,
    background_tasks: BackgroundTasks,
) -> JobCreatedResponse:
    """
    Accept a research topic, register a background worker, and immediately
    return a ``job_id`` without blocking.
    """
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending",
        "topic": body.topic,
        "result": None,
        "error": None,
    }
    background_tasks.add_task(_run_job, job_id, body.topic)
    log.info("Job %s enqueued — topic: %r", job_id, body.topic)
    return JobCreatedResponse(job_id=job_id, status="pending")


@app.get(
    "/research/{job_id}",
    response_model=JobStatusResponse,
    summary="Poll the status of a research job",
)
async def get_research_job(job_id: str) -> JobStatusResponse:
    """
    Return the current status and, once completed, the full Markdown result.
    """
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        topic=job["topic"],
        result=job.get("result"),
        error=job.get("error"),
    )


@app.get("/health", summary="Health check")
async def health() -> dict[str, str]:
    return {"status": "ok"}

