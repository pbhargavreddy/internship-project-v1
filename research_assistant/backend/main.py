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
import threading
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
from fastapi import Path as PathParam  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field, ValidationError  # noqa: E402

from backend.agent import AgentResult, ReActAgent  # noqa: E402
from backend.agent.tools import IngestOutcome, PaperRef, ingest_paper  # noqa: E402
from backend.eval.faithfulness import check_faithfulness  # noqa: E402
from backend.eval.guardrails import detect_injection  # noqa: E402
from backend.agent.memory import SessionStore  # noqa: E402
from backend.ingestion.names import load_doc_names  # noqa: E402
from backend.llm_researcher import run_research_query  # noqa: E402
from backend.llm.router import FinetunedHealth, get_router  # noqa: E402
from backend.retrieval import get_vector_store  # noqa: E402
from backend.retrieval.search import retrieve  # noqa: E402
from backend.storage import documents  # noqa: E402
from backend.storage.chats import (  # noqa: E402
    Chat,
    ChatHistorySessionStore,
    ChatMessage,
    ChatSummary,
    append_message,
    create_chat,
    delete_chat,
    get_chat,
    history_turns,
    is_offline,
    list_chats,
    rename_chat,
)
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

def _startup_documents() -> None:
    """Startup task: migrations (per-chat, Chroma→Mongo, GridFS) and the Chroma rebuild from MongoDB."""
    try:
        summary = documents.startup(get_vector_store())
        log.info(summary.text())
        for err in summary.errors:
            log.warning("Documents startup: %s", err)
    except Exception:  # noqa: BLE001 — a failed migration must not take the API down
        log.exception("Documents startup failed")


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    log.info("Research Assistant API starting up")
    # Background thread: the MongoDB check (up to 3 s) and re-ingestion must not block startup.
    threading.Thread(target=_startup_documents, name="documents-startup", daemon=True).start()
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


UPLOAD_DIR = documents.UPLOAD_DIR
CHUNKS_DIR = documents.CHUNKS_DIR
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


CHAT_ID = r"^[A-Za-z0-9]{1,64}$"


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    skipped: bool = Field(False, description="The chat already had this file")


@app.post(
    "/documents/upload",
    response_model=UploadResponse,
    summary="Upload a PDF, ingest it and return the chunk count",
)
def upload_document(
    file: UploadFile = File(...),
    chat_id: str = Query(..., pattern=CHAT_ID, description="Chat the document belongs to"),
) -> UploadResponse:
    """Index an uploaded PDF for ``chat_id`` (MongoDB + ChromaDB); the file itself is not kept.

    Re-uploading a file the chat already has is skipped; another chat gets its own copy.
    """
    _chat_or_404(chat_id)
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
        try:
            # Extract, chunk, embed → MongoDB + ChromaDB; the PDF file is deleted afterwards.
            added = documents.add_document(tmp_path, doc_id, filename, chat_id, store=get_vector_store())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)
    log.info("Upload %s as %s in chat %s: %s (%d chunks)", filename, doc_id, chat_id, added.status, added.chunk_count)
    return UploadResponse(doc_id=doc_id, filename=filename, chunk_count=added.chunk_count,
                          skipped=added.status == "skipped")


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
    chat_id: str | None = Query(None, pattern=CHAT_ID, description="Only search this chat's documents"),
) -> list[RagHit]:
    """Return the top-k chunks.

    With ``rerank``, fetch ``rerank_candidates`` (default 10) from ChromaDB and
    reorder them with the cross-encoder; scores are then reranker logits, not
    cosine similarities. Falls back to vector order if no model is trained.
    """
    if not q.strip():
        raise HTTPException(status_code=422, detail="q must not be blank")
    hits = retrieve(q, k=k, doc_ids=doc_ids, rerank=rerank, chat_id=chat_id)
    return [RagHit(**chunk.model_dump(), score=score) for chunk, score in hits]


class ChatRequest(BaseModel):
    chat_id: str = Field(..., pattern=CHAT_ID, description="Chat from POST /chats")
    message: str = Field(..., min_length=1, max_length=4000)


class ChatName(BaseModel):
    name: str = Field(..., min_length=1, max_length=200, description="1-60 characters after trimming")


class StorageStatus(BaseModel):
    online: bool
    mode: str = Field(..., description="mongodb | local")


def _chat_or_404(chat_id: str) -> Chat:
    chat = get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found")
    return chat


@app.post("/chats", response_model=ChatSummary, status_code=201, summary="Create a named chat")
def chats_create(body: ChatName) -> ChatSummary:
    try:
        return create_chat(body.name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/chats", response_model=list[ChatSummary], summary="List chats, most recently updated first")
def chats_list() -> list[ChatSummary]:
    return list_chats()


@app.get("/chats/{chat_id}", response_model=Chat, summary="One chat with its messages")
def chats_get(chat_id: str = PathParam(..., pattern=CHAT_ID)) -> Chat:
    return _chat_or_404(chat_id)


@app.patch("/chats/{chat_id}", response_model=ChatSummary, summary="Rename a chat")
def chats_rename(body: ChatName, chat_id: str = PathParam(..., pattern=CHAT_ID)) -> ChatSummary:
    try:
        chat = rename_chat(chat_id, body.name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if chat is None:
        raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found")
    return chat


@app.delete("/chats/{chat_id}", status_code=204, summary="Delete a chat and its documents")
def chats_delete(chat_id: str = PathParam(..., pattern=CHAT_ID)) -> None:
    """Delete the chat, its documents (shared files survive while another chat uses them) and its session."""
    _chat_or_404(chat_id)
    removed = documents.remove_chat_documents(chat_id, store=get_vector_store(),
                                              uploads_dir=UPLOAD_DIR, chunks_dir=CHUNKS_DIR)
    if not delete_chat(chat_id):
        raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found")
    SessionStore().delete(chat_id)  # its Doc N numbering / sources
    log.info("Deleted chat %s and %d document(s)", chat_id, len(removed))


@app.get("/health/storage", response_model=StorageStatus, summary="Where chats are being saved")
def health_storage() -> StorageStatus:
    offline = is_offline()
    return StorageStatus(online=not offline, mode="local" if offline else "mongodb")


@app.post("/agent/chat", response_model=AgentResult, summary="Ask the ReAct research agent")
def agent_chat(body: ChatRequest) -> AgentResult:  # sync def: LLM + tools block, so run in threadpool
    """Run one agent turn in a stored chat and save both the user and the agent message.

    The prompt history is the chat's last ``agent_history_turns`` stored exchanges.
    """
    if not body.message.strip():
        raise HTTPException(status_code=422, detail="message must not be blank")
    phrase = detect_injection(body.message)
    if phrase is not None:
        log.warning("Rejected prompt-injection attempt in chat %s: %r", body.chat_id, phrase)
        raise HTTPException(
            status_code=400,
            detail=f"Message rejected by the prompt-injection filter (matched: \"{phrase}\"). "
            "Please rephrase your question.",
        )
    chat = _chat_or_404(body.chat_id)
    turns = history_turns(chat, get_settings().agent_history_turns)
    try:
        agent = ReActAgent(store=ChatHistorySessionStore(turns))
    except EnvironmentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    append_message(body.chat_id, ChatMessage(role="user", content=body.message))
    result = agent.chat(body.chat_id, body.message)
    if result.cited_chunks:  # nothing to check against if the agent retrieved no passages
        result.faithfulness = check_faithfulness(result.answer, result.cited_chunks)
        if result.faithfulness.score < get_settings().faithfulness_warn_below:
            result.answer += (
                f"\n\n> ⚠️ Low faithfulness ({result.faithfulness.score:.0%}): "
                f"{len(result.faithfulness.unsupported)} sentence(s) are not well supported by the cited passages."
            )
    append_message(body.chat_id, ChatMessage.from_result(result))
    return result


@app.post("/papers/ingest", response_model=IngestOutcome, summary="Download and index one paper_search result")
def papers_ingest(
    paper: PaperRef, chat_id: str = Query(..., pattern=CHAT_ID, description="Chat the paper is added to")
) -> IngestOutcome:
    """Ingest a paper from an agent response's ``papers`` list into ``chat_id`` (first ``ingest_max_pages`` pages).

    Returns ``status`` ingested / already_ingested / failed (with ``detail``); a failed
    download is reported in the body, not as an HTTP error.
    """
    try:
        full = Paper(**paper.model_dump())
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _chat_or_404(chat_id)
    return ingest_paper(full, chat_id)


@app.delete("/documents/{doc_id}", response_model=documents.DocRemoval, summary="Remove a document from a chat")
def delete_document(
    doc_id: str = PathParam(..., pattern=r"^[0-9a-f]{16}$"),
    chat_id: str = Query(..., pattern=CHAT_ID, description="Chat to remove it from"),
) -> documents.DocRemoval:
    """Remove a document from one chat: MongoDB (record + chunks), ChromaDB and names.json.

    Other chats holding the same file keep their copies. Earlier answers keep their citation text.
    """
    removal = documents.remove_document(doc_id, chat_id, store=get_vector_store(),
                                        uploads_dir=UPLOAD_DIR, chunks_dir=CHUNKS_DIR)
    if not removal.found:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} is not in chat {chat_id}")
    return removal


class DocumentInfo(BaseModel):
    doc_id: str
    title: str
    chunk_count: int


@app.get("/documents", response_model=list[DocumentInfo], summary="List a chat's documents")
def list_documents(chat_id: str = Query(..., pattern=CHAT_ID, description="Chat whose documents to list")) -> list[DocumentInfo]:
    """The chat's PDFs (uploads and ingested papers) by name; ``title`` is the filename used in this chat."""
    store = get_vector_store()
    names = load_doc_names(chat_id=chat_id)
    ids = documents.chat_doc_ids(chat_id, store)
    docs = [DocumentInfo(doc_id=d, title=names.get(d, f"{d}.pdf"), chunk_count=store.doc_chunk_count(d, chat_id))
            for d in ids]
    return sorted(docs, key=lambda d: d.title.casefold())


@app.get("/health", summary="Health check")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/finetuned", response_model=FinetunedHealth, summary="Fine-tuned (Kaggle) endpoint status")
def health_finetuned() -> FinetunedHealth:
    """Cached (60 s) health of ``FINETUNED_BASE_URL`` and whether its circuit breaker is open."""
    return get_router().health()

