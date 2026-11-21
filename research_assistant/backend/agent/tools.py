"""
tools.py
--------
Tools the ReAct agent can call. Each has a ``name``, a ``description``, a
pydantic ``args_model`` (its JSON arguments) and ``run(args) -> str``.

  • rag_search         → reranked chunks from the local vector store
  • paper_search       → arXiv + Semantic Scholar, optionally ingesting PDFs
  • summarize_papers   → TLDR of retrieved chunks by the fine-tuned model (API LLM fallback)
  • python_interpreter → runs Python in a subprocess (NOT a security sandbox)

Tools that call a model record its label in ``model_used`` after each run.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
import contextvars
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any, ClassVar

from pydantic import BaseModel, BeforeValidator, Field

from backend.agent.memory import Citation, Session
from backend.config import get_settings
from backend.eval.guardrails import sanitize_text
from backend.ingestion import Chunk
from backend.ingestion.names import is_known_doc_name, load_doc_names
from backend.llm.router import SYSTEM_PROMPT, FinetunedRouter, api_llm_label, get_router
from backend.runtime import timed

log = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 4000
ABSTRACT_CHARS = 300

RetrieveFn = Callable[[str, int], list[tuple[Chunk, float]]]  # (query, k), already scoped to the chat
DocRetrieveFn = Callable[[str, int, list[str] | None], list[tuple[Chunk, float]]]
ResolveDocFn = Callable[[str], str | None]  # doc name / doc_id -> stored doc_id
DocChunksFn = Callable[[str], list[Chunk]]  # doc name / doc_id -> its chunks in page order

K_MIN, K_MAX = 1, 10
_DOC_NUMBER_RE = re.compile(r"^(?:doc\s*)?(\d+)$", re.I)


def _clamp_k(value: Any) -> Any:
    """Clamp a requested result count into [1, 10] instead of rejecting it."""
    try:
        return max(K_MIN, min(K_MAX, int(value)))
    except (TypeError, ValueError):
        return value  # not a number: let pydantic report it


ClampedK = Annotated[int, BeforeValidator(_clamp_k)]


def session_doc_ref(session: Session, ref: str) -> str:
    """Turn ``"Doc 2"`` / ``"2"`` into that session document's doc_id; other refs pass through."""
    m = _DOC_NUMBER_RE.match(ref.strip())
    return (session.doc_id_for(int(m.group(1))) or ref.strip()) if m else ref.strip()


def unknown_doc_message(doc: str, chat_id: str | None = None) -> str:
    names = sorted(set(load_doc_names(chat_id=chat_id).values()))
    known = ", ".join(names[:20]) + (" ..." if len(names) > 20 else "")
    return f"Error: no document named '{doc}' in this chat. Its documents: {known or 'none'}."


def _store_resolve(doc: str, chat_id: str | None) -> str | None:
    from backend.retrieval import get_vector_store  # heavy: chromadb

    return get_vector_store().resolve_doc_id(doc, chat_id=chat_id)


def _store_doc_chunks(doc: str, chat_id: str | None) -> list[Chunk]:
    from backend.retrieval import get_vector_store  # heavy: chromadb

    return get_vector_store().get_doc_chunks(doc, chat_id=chat_id)


def truncate(text: str, limit: int) -> str:
    """Cut ``text`` to ``limit`` characters, marking the cut."""
    return text if len(text) <= limit else text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


class Tool(ABC):
    """Base class for agent tools."""

    name: ClassVar[str]
    description: ClassVar[str]
    args_model: ClassVar[type[BaseModel]]
    model_used: str | None = None  # label of the model the last run called, if any

    def signature(self) -> str:
        """Compact JSON-args description for the system prompt, e.g. ``{"query": str, "k": int=5}``."""
        parts = []
        for field, info in self.args_model.model_fields.items():
            ann = getattr(info.annotation, "__name__", str(info.annotation))
            default = "" if info.is_required() else f"={info.default!r}"
            parts.append(f'"{field}": {ann}{default}')
        return "{" + ", ".join(parts) + "}"

    def __call__(self, args: dict[str, Any]) -> str:
        """Validate ``args`` and run the tool.

        Raises:
            pydantic.ValidationError: If ``args`` don't match ``args_model``.
        """
        return self.run(self.args_model.model_validate(args))

    @abstractmethod
    def run(self, args: Any) -> str:
        """Execute with validated arguments and return the observation text."""


# ── rag_search ───────────────────────────────────────────────────────────────


class RagSearchArgs(BaseModel):
    query: str = Field(..., min_length=1)
    k: ClampedK = 5
    doc: str | None = None


def _chat_retrieve(chat_id: str) -> DocRetrieveFn:
    """Reranked retrieval limited to ``chat_id``'s documents."""

    def run(query: str, k: int, doc_ids: list[str] | None = None) -> list[tuple[Chunk, float]]:
        from backend.retrieval.search import retrieve  # heavy: chromadb + torch

        return retrieve(query, k=k, doc_ids=doc_ids, rerank=True, chat_id=chat_id)

    return run


GENERIC_DOC_QUERY = "main contribution, method and results"


def strip_doc_name(query: str, doc: str) -> str:
    """Drop the filename (and its stem) from a doc-filtered query: the filter already picks the
    document, and "R6.pdf" in the text only pulls the embedding away from the content asked about.

    Example: ``"What dataset did the paper R6.pdf use?"`` → ``"What dataset did the paper use?"``.
    """
    names = sorted({doc, Path(doc).stem} - {""}, key=len, reverse=True)
    if not names or doc.lower().startswith("doc "):  # "Doc 3" references, not filenames
        return query
    pattern = r"(?<!\w)(" + "|".join(re.escape(n) for n in names) + r")(?!\w)"
    cleaned = re.sub(r"\s{2,}", " ", re.sub(pattern, "", query, flags=re.IGNORECASE)).strip()
    cleaned = re.sub(r"\s+([?.!,])", r"\1", cleaned)
    return cleaned if re.search(r"\w{3,}", cleaned) else GENERIC_DOC_QUERY


class RagSearchTool(Tool):
    """Searches ingested papers and numbers documents stably within the session.

    Args:
        session: Conversation whose ``Doc N`` numbering and sources are updated.
        retrieve_fn: ``(query, k) -> [(chunk, score)]``; defaults to reranked retrieval.
        doc_retrieve_fn: ``(query, k, doc_ids)``, used when ``doc`` is given.
        resolve_doc_fn: Maps a filename / doc_id to a stored doc_id. Defaults to the vector store.
    """

    name = "rag_search"
    description = (
        "Search the locally ingested/uploaded papers. Optional doc (filename like \"R6.pdf\" or "
        "\"Doc N\") limits the search to that file. Returns passages labelled [Doc N, Sec S, p.P]."
    )
    args_model = RagSearchArgs

    def __init__(
        self,
        session: Session,
        retrieve_fn: RetrieveFn | None = None,
        doc_retrieve_fn: DocRetrieveFn | None = None,
        resolve_doc_fn: ResolveDocFn | None = None,
    ) -> None:
        self.session = session
        chat_id = session.session_id  # one session per chat: only this chat's documents are visible
        self.chat_id: str = chat_id
        self.retrieve_fn: RetrieveFn = retrieve_fn or _chat_retrieve(chat_id)
        self.doc_retrieve_fn: DocRetrieveFn = doc_retrieve_fn or _chat_retrieve(chat_id)
        self.resolve_doc_fn: ResolveDocFn = resolve_doc_fn or (lambda ref: _store_resolve(ref, chat_id))
        self.retrieved: list[tuple[int, Chunk]] = []  # (Doc N, chunk) seen this turn

    def run(self, args: RagSearchArgs) -> str:
        doc, query = args.doc, args.query
        if not doc and is_known_doc_name(query, chat_id=self.chat_id):
            # The model searched for a filename; a semantic search for "R6.pdf" finds nothing useful.
            doc, query = query, GENERIC_DOC_QUERY
        if doc:
            doc_id = self.resolve_doc_fn(session_doc_ref(self.session, doc))
            if doc_id is None:
                return unknown_doc_message(doc, self.chat_id)
            hits = self.doc_retrieve_fn(strip_doc_name(query, doc), args.k, [doc_id])
        else:
            hits = self.retrieve_fn(query, args.k)
        if not hits:
            return "No matching passages in local documents. Try paper_search with ingest=true."
        return self.format_hits(hits)

    def format_hits(self, hits: list[tuple[Chunk, float]]) -> str:
        """Number docs, record citable sources and render hits as ``[Doc N, Sec S, p.P] text``."""
        lines = []
        for chunk, _score in hits:
            n = self.session.doc_number(chunk.doc_id)
            section = chunk.section or "n/a"
            self.session.add_source(
                Citation(doc=n, title=self.session.title_for(chunk.doc_id), page=chunk.page, section=chunk.section)
            )
            self.retrieved.append((n, chunk))
            # Strip injected instructions hidden in documents before the LLM sees them.
            lines.append(f"[Doc {n}, Sec {section}, p.{chunk.page}] {sanitize_text(chunk.text).strip()}")
        return "\n\n".join(lines)


# ── paper_search ─────────────────────────────────────────────────────────────


class PaperSearchArgs(BaseModel):
    query: str = Field(..., min_length=1)
    max_results: int = Field(5, ge=1, le=10)
    ingest: bool = False


class PaperRef(BaseModel):
    """A paper found by paper_search, as returned to clients (and accepted by ``POST /papers/ingest``)."""

    id: str = Field(..., description="Source-prefixed ID, e.g. arxiv:1706.03762")
    title: str
    year: int | None = None
    authors: list[str] = Field(default_factory=list)
    url: str
    pdf_url: str | None = None
    source: str
    citation_count: int | None = None


class IngestOutcome(BaseModel):
    """Result of ingesting one paper's PDF."""

    paper_id: str
    title: str
    status: str = Field(..., description="ingested | already_ingested | failed")
    doc_id: str | None = None
    chunk_count: int | None = None
    detail: str | None = None


def ingest_paper(paper: Any, chat_id: str) -> IngestOutcome:
    """Download and index one paper (first ``ingest_max_pages`` pages) for ``chat_id``; never raises.

    A paper the chat already holds is not re-chunked or re-embedded. The title is
    registered as the document's name in that chat.

    Args:
        paper: A ``backend.tools.web_search.Paper``.
        chat_id: Chat that gets the document.

    Returns:
        What happened, with the ``doc_id`` on success.
    """
    from backend.ingestion import compute_doc_id
    from backend.retrieval import get_vector_store
    from backend.storage import documents
    from backend.tools.web_search import download_pdf

    out = IngestOutcome(paper_id=paper.id, title=paper.title, status="failed")
    try:
        if not paper.pdf_url:
            out.detail = "no PDF link"
            return out
        with timed("download", paper=paper.id):
            path = download_pdf(paper)
        if path is None:
            out.detail = "PDF download failed"
            return out
        doc_id = compute_doc_id(path)
        # The downloaded PDF is deleted afterwards; chunks + embeddings live in MongoDB / ChromaDB.
        added = documents.add_document(path, doc_id, paper.title, chat_id, store=get_vector_store(),
                                       title=paper.title, max_pages=get_settings().ingest_max_pages)
        out.status = "already_ingested" if added.status == "skipped" else "ingested"
        out.chunk_count, out.doc_id = added.chunk_count, doc_id
    except Exception as exc:  # noqa: BLE001 — one bad PDF must not stop the rest
        log.warning("Ingest failed for %s: %s", paper.id, exc)
        out.status, out.detail = "failed", str(exc)
    return out


class PaperSearchTool(Tool):
    """Searches arXiv / Semantic Scholar; with ``ingest`` also indexes the top PDFs.

    Results are kept in :attr:`papers` (structured) so the agent can return them.

    Args:
        session: Conversation that receives ``Doc N`` numbers and titles for ingested papers.
    """

    name = "paper_search"
    description = (
        "Search arXiv and Semantic Scholar for papers. Returns titles with links. With ingest=true, the top "
        "PDFs are downloaded and indexed so rag_search can read their full text."
    )
    args_model = PaperSearchArgs

    def __init__(self, session: Session) -> None:
        self.session = session
        self.papers: list[PaperRef] = []  # every paper found this turn

    def run(self, args: PaperSearchArgs) -> str:
        from backend.tools.web_search import SearchUnavailableError, search_papers

        try:
            papers = search_papers(args.query, max_results=args.max_results)
        except SearchUnavailableError as exc:
            return f"Error: paper search unavailable ({exc})"
        if not papers:
            return "No papers found."
        lines = []
        for i, p in enumerate(papers, 1):
            ref = PaperRef(**p.model_dump(include=set(PaperRef.model_fields)))
            if all(r.id != ref.id for r in self.papers):
                self.papers.append(ref)
            authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
            pdf = f" | PDF: {p.pdf_url}" if p.pdf_url else ""
            cites = f" | citations: {p.citation_count}" if p.citation_count is not None else ""
            lines.append(
                f"{i}. [{p.title}]({p.url}) ({p.year or 'n.d.'}) — {authors or 'unknown authors'}{pdf}{cites}\n"
                f"   {truncate(p.abstract, ABSTRACT_CHARS)}"
            )
        if args.ingest:
            lines.append(self._ingest(papers[: get_settings().ingest_max_papers]))
        return "\n".join(lines)

    def _ingest(self, papers: list[Any]) -> str:
        """Ingest ``papers`` and give each a ``Doc N`` in this session."""
        done, failed = [], []
        for paper in papers:
            outcome = ingest_paper(paper, self.session.session_id)
            if outcome.doc_id is None:
                failed.append(paper.title)
                continue
            self.session.doc_titles[outcome.doc_id] = paper.title
            done.append(f"Doc {self.session.doc_number(outcome.doc_id)}: {paper.title}")
        msg = "Ingested (now searchable with rag_search): " + ("; ".join(done) or "none")
        return msg + (f"\nCould not ingest: {'; '.join(failed)}" if failed else "")


# ── summarize_papers ─────────────────────────────────────────────────────────

SUMMARY_MAX_TOKENS = 300
MAX_SECTION_CHUNKS = 3
SECTION_CHAR_BUDGET = 6000  # ≈1,500 tokens of context per section call
MAX_SECTION_WORKERS = 5
FINETUNED_SHORT = "Fine-tuned Qwen"
SECTION_HEADINGS = ["Objective", "Method", "Dataset/Experiments", "Results", "Conclusion"]
# Checked in this order, so "Experimental Results" → Results and "Results and Conclusion" → Conclusion.
_SECTION_PATTERNS: list[tuple[int, re.Pattern[str]]] = [
    (0, re.compile(r"\b(abstract|introduction|motivation|overview|background)\b", re.I)),
    (4, re.compile(r"\b(conclusions?|concluding|future work|limitations?|summary)\b", re.I)),
    (3, re.compile(r"\b(results?|comparisons?|ablations?|performance|discussion|analysis)\b", re.I)),
    (2, re.compile(r"\b(datasets?|data|experiments?|experimental|setup|implementation|training|"
                   r"benchmarks?|evaluation|metrics?)\b", re.I)),
    (1, re.compile(r"\b(methods?|methodology|approach|proposed|architecture|framework|model|"
                   r"algorithm|network)\b", re.I)),
]
_CITE_MARK_RE = re.compile(r"\[Doc\s*\d+", re.I)


def _section_group(section: str | None) -> int | None:
    if not section:
        return None
    return next((g for g, pattern in _SECTION_PATTERNS if pattern.search(section)), None)


def group_sections(chunks: list[Chunk]) -> list[tuple[str, list[Chunk]]]:
    """Split a document's chunks (page order) into up to 5 summary sections.

    Chunks under a recognised heading start that group; chunks under other headings
    (subsections, dataset names, ...) stay in the current group. If no heading is
    recognised, the chunks are split into 5 equal parts in page order.

    Returns:
        ``(heading, chunks)`` pairs in :data:`SECTION_HEADINGS` order, empty groups dropped.
    """
    groups: list[list[Chunk]] = [[] for _ in SECTION_HEADINGS]
    labels = [_section_group(c.section) for c in chunks]
    if any(label is not None for label in labels):
        current = 0
        for chunk, label in zip(chunks, labels):
            current = label if label is not None else current
            groups[current].append(chunk)
    else:
        n = len(chunks)
        for i, chunk in enumerate(chunks):
            groups[i * len(SECTION_HEADINGS) // n].append(chunk)
    return [(h, g) for h, g in zip(SECTION_HEADINGS, groups) if g]


def _pick(chunks: list[Chunk], n: int = MAX_SECTION_CHUNKS) -> list[Chunk]:
    """Up to ``n`` chunks spread evenly over the group (first, middle, last)."""
    if len(chunks) <= n:
        return chunks
    return [chunks[round(i * (len(chunks) - 1) / (n - 1))] for i in range(n)]


class SummarizePapersArgs(BaseModel):
    doc: str = Field(..., min_length=1, description="Filename (e.g. R6.pdf), paper title, doc_id or Doc N")


class SectionSummary(BaseModel):
    heading: str
    text: str
    finetuned: bool
    fallback_reason: str | None = None


class SummarizePapersTool(Tool):
    """Summarizes one document section by section with the fine-tuned model.

    Up to 5 sections are summarized concurrently through the shared router; each
    section falls back to the API LLM on its own if the fine-tuned call fails.

    Args:
        session: Conversation whose ``Doc N`` numbering and sources are updated.
        doc_chunks_fn: ``doc -> chunks in page order``. Defaults to the vector store's ``get_doc_chunks``.
        router: Fine-tuned endpoint router. Defaults to :func:`get_router`.
        fallback_llm: Fallback chat model. Defaults to :func:`backend.llm.get_llm`.
    """

    name = "summarize_papers"
    description = (
        "Summarize ONE document section by section (objective, method, data, results, conclusion). "
        "Use for 'summarize <file>' / 'what is <file> about'. doc = filename like \"R6.pdf\", or \"Doc N\"."
    )
    args_model = SummarizePapersArgs

    def __init__(
        self,
        session: Session,
        doc_chunks_fn: DocChunksFn | None = None,
        router: FinetunedRouter | None = None,
        fallback_llm: Any | None = None,
    ) -> None:
        self.session = session
        self.doc_chunks_fn: DocChunksFn = doc_chunks_fn or (lambda ref: _store_doc_chunks(ref, session.session_id))
        self.rag = RagSearchTool(session, retrieve_fn=lambda q, k: [])  # for numbering/citations only
        self.router: FinetunedRouter = router or get_router()
        self._fallback_llm = fallback_llm
        self.last_summary: str | None = None  # combined markdown of the last successful run

    @property
    def retrieved(self) -> list[tuple[int, Chunk]]:
        """Chunks seen this turn (read by the agent for the faithfulness check)."""
        return self.rag.retrieved

    def _fallback(self) -> Any:
        if self._fallback_llm is None:
            from backend.llm import get_llm

            self._fallback_llm = get_llm()
        return self._fallback_llm

    def _summarize_section(self, heading: str, chunks: list[Chunk]) -> SectionSummary:
        budget = SECTION_CHAR_BUDGET // len(chunks)
        labels = [f"[Doc {self.session.doc_number(c.doc_id)}, Sec {c.section or 'n/a'}]" for c in chunks]
        context = "\n\n".join(f"{label} {sanitize_text(c.text).strip()[:budget]}" for label, c in zip(labels, chunks))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Summarize the {heading.lower()} of this paper in 2-3 sentences. "
                f"Cite the sections you used as [Doc N, Sec S].\n\n{context}"
            )},
        ]
        result = self.router.complete(messages, max_tokens=SUMMARY_MAX_TOKENS, temperature=0.0)
        text, finetuned = result.text, result.text is not None
        if text is None:
            try:
                text = self._fallback().generate(messages, max_tokens=SUMMARY_MAX_TOKENS, temperature=0.0).strip()
            except Exception as exc:  # noqa: BLE001 — one failed section must not sink the summary
                log.warning("Section %s fallback failed: %s", heading, exc)
                text = f"(summary unavailable: {type(exc).__name__})"
        if not _CITE_MARK_RE.search(text):  # keep this section's citations even if the model dropped them
            text = f"{text} {' '.join(dict.fromkeys(labels))}"
        return SectionSummary(heading=heading, text=text, finetuned=finetuned, fallback_reason=result.fallback_reason)

    def _model_label(self, parts: list[SectionSummary]) -> str:
        n, k = len(parts), sum(p.finetuned for p in parts)
        if k == n:
            return f"{FINETUNED_SHORT} ({k}/{n} sections)"
        if k == 0:
            return api_llm_label()
        return f"{FINETUNED_SHORT} ({k}/{n} sections) + {api_llm_label()} ({n - k}/{n})"

    def run(self, args: SummarizePapersArgs) -> str:
        self.model_used, self.last_summary = None, None
        chunks = self.doc_chunks_fn(session_doc_ref(self.session, args.doc))
        if not chunks:
            return unknown_doc_message(args.doc, self.session.session_id)
        doc_id = chunks[0].doc_id
        n = self.session.doc_number(doc_id)
        groups = [(heading, _pick(group)) for heading, group in group_sections(chunks)]
        self.rag.format_hits([(c, 0.0) for _, picked in groups for c in picked])  # sources + retrieved

        with timed("summarize", doc_id=doc_id, sections=len(groups)):
            with ThreadPoolExecutor(max_workers=min(MAX_SECTION_WORKERS, len(groups))) as pool:
                ctx = contextvars.copy_context()  # keeps the per-turn LLM usage counter
                parts = list(pool.map(lambda g: ctx.copy().run(self._summarize_section, *g), groups))

        self.model_used = self._model_label(parts)
        body = "\n\n".join(f"### {p.heading}\n{p.text}" for p in parts)
        self.last_summary = f"**Summary of {self.session.title_for(doc_id)}** (Doc {n})\n\n{body}"
        return self.last_summary


# ── python_interpreter ───────────────────────────────────────────────────────


class PythonArgs(BaseModel):
    code: str = Field(..., min_length=1)


def run_python(code: str, timeout: float | None = None, output_dir: str | Path | None = None) -> str:
    """Run ``code`` with the server's (venv) Python in a fresh temp directory.

    This is NOT a security sandbox: the code runs as the server user, can read
    and write files and open network connections. It only isolates the working
    directory and environment, and limits run time and output size.

    Args:
        code: Python source.
        timeout: Seconds before the process group is killed. Defaults to ``settings.python_timeout_seconds``.
        output_dir: Where saved ``.png`` files are copied. Defaults to ``settings.agent_output_dir``.

    Returns:
        Exit status, stdout and stderr (truncated to 4,000 chars), plus paths of saved PNGs.
    """
    settings = get_settings()
    timeout = timeout if timeout is not None else settings.python_timeout_seconds
    out_dir = Path(output_dir) if output_dir is not None else settings.agent_output_dir
    with tempfile.TemporaryDirectory(prefix="agent_py_") as tmp:
        script = Path(tmp) / "main.py"
        script.write_text(code, encoding="utf-8")
        env = {"PATH": os.defpath, "HOME": tmp, "MPLBACKEND": "Agg", "MPLCONFIGDIR": tmp}
        proc = subprocess.Popen(
            [sys.executable, "-I", str(script)],
            cwd=tmp,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,  # own process group, so children die with it
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            status = f"exit code {proc.returncode}"
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
            status = f"Error: timed out after {timeout:g} s (process killed)"
        saved = []
        for png in sorted(Path(tmp).rglob("*.png")):
            out_dir.mkdir(parents=True, exist_ok=True)
            dest = out_dir / f"{uuid.uuid4().hex[:8]}_{png.name}"
            shutil.copy2(png, dest)
            saved.append(str(dest.resolve()))
    text = f"{status}\n--- stdout ---\n{stdout}"
    if stderr:
        text += f"\n--- stderr ---\n{stderr}"
    text = truncate(text, MAX_OUTPUT_CHARS)
    if saved:
        text += "\nSaved figures: " + ", ".join(saved)
    return text


class PythonInterpreterTool(Tool):
    """Executes Python for arithmetic, statistics and plots (see :func:`run_python`)."""

    name = "python_interpreter"
    description = (
        "Run a Python script (15 s limit, no input). Use print() to show results; "
        "numpy/pandas/matplotlib may be available. Saved .png files are returned as paths."
    )
    args_model = PythonArgs

    def run(self, args: PythonArgs) -> str:
        return run_python(args.code)


def default_tools(session: Session) -> list[Tool]:
    """The standard tool set bound to ``session``."""
    return [RagSearchTool(session), PaperSearchTool(session), SummarizePapersTool(session), PythonInterpreterTool()]
