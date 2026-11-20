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

import hashlib
import shutil
import tempfile
from pathlib import Path

from backend.config import get_settings
from backend.runtime import limit_cpu_threads

# Before torch / numpy / sentence-transformers are imported, so the env caps apply.
limit_cpu_threads(get_settings().cpu_threads)

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field, ValidationError  # noqa: E402

from backend.agent import AgentResult, ReActAgent  # noqa: E402
from backend.agent.tools import IngestOutcome, PaperRef, ingest_paper  # noqa: E402
from backend.eval.faithfulness import check_faithfulness  # noqa: E402
from backend.eval.guardrails import detect_injection  # noqa: E402
from backend.ingestion import ingest_pdf  # noqa: E402
from backend.ingestion.names import load_doc_names, save_doc_names  # noqa: E402
from backend.llm_researcher import run_research_query  # noqa: E402
from backend.llm.router import FinetunedHealth, get_router  # noqa: E402
from backend.retrieval import get_vector_store  # noqa: E402
from backend.retrieval.search import retrieve  # noqa: E402
from backend.tools.web_search import (  # noqa: E402
    ALL_SOURCES,
    Paper,
    SearchUnavailableError,
    Source,
    search_papers,
)

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

# python_interpreter plots, copied out of the temp dir by run_python; GET /plots/<file>.
get_settings().agent_output_dir.mkdir(parents=True, exist_ok=True)
app.mount("/plots", StaticFiles(directory=get_settings().agent_output_dir), name="plots")


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


@app.get(
    "/search",
    response_model=list[Paper],
    summary="Search arXiv and Semantic Scholar for papers",
)
def search(  # sync def: FastAPI runs it in a threadpool, so blocking I/O is fine
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    max_results: int = Query(10, ge=1, le=50, description="Maximum merged results"),
    sources: list[Source] = Query(list(ALL_SOURCES), description="Sources to query"),
) -> list[Paper]:
    """Return deduplicated papers. Responds 503 only if every source is down."""
    if not q.strip():
        raise HTTPException(status_code=422, detail="q must not be blank")
    try:
        return search_papers(q, max_results=max_results, sources=sources)
    except SearchUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


UPLOAD_DIR = Path("data/uploads")
CHUNKS_DIR = Path("data/chunks")
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int


@app.post(
    "/documents/upload",
    response_model=UploadResponse,
    summary="Upload a PDF, ingest it and return the chunk count",
)
def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """Store the PDF under data/uploads/<doc_id>.pdf and its chunks as JSONL."""
    filename = file.filename or "upload.pdf"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=UPLOAD_DIR, suffix=".part", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        if tmp_path.stat().st_size > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="PDF larger than 50 MB")
        with tmp_path.open("rb") as fh:
            if fh.read(5) != b"%PDF-":
                raise HTTPException(status_code=400, detail=f"{filename} is not a PDF")
        doc_id = hashlib.sha256(tmp_path.read_bytes()).hexdigest()[:16]
        stored = tmp_path.replace(UPLOAD_DIR / f"{doc_id}.pdf")
        try:
            chunks = ingest_pdf(stored, out_dir=CHUNKS_DIR)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)
    get_vector_store().add_chunks(chunks)
    save_doc_names({doc_id: filename})
    log.info("Uploaded %s as %s (%d chunks indexed)", filename, doc_id, len(chunks))
    return UploadResponse(doc_id=doc_id, filename=filename, chunk_count=len(chunks))


class RagHit(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    page: int
    section: str | None
    score: float


@app.get(
    "/rag/search",
    response_model=list[RagHit],
    summary="Semantic search over uploaded document chunks",
)
def rag_search(
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    k: int = Query(5, ge=1, le=50, description="Number of chunks to return"),
    doc_ids: list[str] | None = Query(None, description="Restrict to these doc_ids"),
    rerank: bool = Query(True, description="Rerank vector hits with the cross-encoder"),
) -> list[RagHit]:
    """Return the top-k chunks.

    With ``rerank``, fetch ``rerank_candidates`` (default 10) from ChromaDB and
    reorder them with the cross-encoder; scores are then reranker logits, not
    cosine similarities. Falls back to vector order if no model is trained.
    """
    if not q.strip():
        raise HTTPException(status_code=422, detail="q must not be blank")
    hits = retrieve(q, k=k, doc_ids=doc_ids, rerank=rerank)
    return [RagHit(**chunk.model_dump(), score=score) for chunk, score in hits]


class ChatRequest(BaseModel):
    session_id: str = Field(..., pattern=r"^[A-Za-z0-9_-]{1,64}$", description="Conversation ID")
    message: str = Field(..., min_length=1, max_length=4000)


_agent: ReActAgent | None = None


@app.post("/agent/chat", response_model=AgentResult, summary="Ask the ReAct research agent")
def agent_chat(body: ChatRequest) -> AgentResult:  # sync def: LLM + tools block, so run in threadpool
    """Run one agent turn; memory is kept per ``session_id`` under data/sessions/."""
    global _agent
    if not body.message.strip():
        raise HTTPException(status_code=422, detail="message must not be blank")
    phrase = detect_injection(body.message)
    if phrase is not None:
        log.warning("Rejected prompt-injection attempt in session %s: %r", body.session_id, phrase)
        raise HTTPException(
            status_code=400,
            detail=f"Message rejected by the prompt-injection filter (matched: \"{phrase}\"). "
            "Please rephrase your question.",
        )
    try:
        if _agent is None:
            _agent = ReActAgent()
        result = _agent.chat(body.session_id, body.message)
    except EnvironmentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if result.cited_chunks:  # nothing to check against if the agent retrieved no passages
        result.faithfulness = check_faithfulness(result.answer, result.cited_chunks)
        if result.faithfulness.score < get_settings().faithfulness_warn_below:
            result.answer += (
                f"\n\n> ⚠️ Low faithfulness ({result.faithfulness.score:.0%}): "
                f"{len(result.faithfulness.unsupported)} sentence(s) are not well supported by the cited passages."
            )
    return result


@app.post("/papers/ingest", response_model=IngestOutcome, summary="Download and index one paper_search result")
def papers_ingest(paper: PaperRef) -> IngestOutcome:
    """Ingest a paper from an agent response's ``papers`` list (first ``ingest_max_pages`` pages).

    Returns ``status`` ingested / already_ingested / failed (with ``detail``); a failed
    download is reported in the body, not as an HTTP error.
    """
    try:
        full = Paper(**paper.model_dump())
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ingest_paper(full)


class DocumentInfo(BaseModel):
    doc_id: str
    title: str
    chunk_count: int


@app.get("/documents", response_model=list[DocumentInfo], summary="List uploaded documents")
def list_documents() -> list[DocumentInfo]:
    """Uploaded PDFs, newest first; ``title`` is the original filename when known."""
    names = load_doc_names()
    docs = []
    for pdf in sorted(UPLOAD_DIR.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True):
        chunks = CHUNKS_DIR / f"{pdf.stem}.jsonl"
        count = sum(1 for _ in chunks.open(encoding="utf-8")) if chunks.exists() else 0
        docs.append(DocumentInfo(doc_id=pdf.stem, title=names.get(pdf.stem, pdf.name), chunk_count=count))
    return docs


@app.get("/health", summary="Health check")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/finetuned", response_model=FinetunedHealth, summary="Fine-tuned (Kaggle) endpoint status")
def health_finetuned() -> FinetunedHealth:
    """Cached (60 s) health of ``FINETUNED_BASE_URL`` and whether its circuit breaker is open."""
    return get_router().health()

