# ScholarPulse

An autonomous AI research assistant. A FastAPI backend runs LLM research jobs, and a Streamlit frontend submits topics and shows the Markdown reports. The work is planned in sprints (see `Prompts/sprints.txt`): ingestion and retrieval, then a PyTorch reranker, QLoRA fine-tuning, a ReAct agent with tools, evaluation and guardrails, and streaming API serving with telemetry.

## Repository layout

All application code lives under `research_assistant/`. The git root holds only supporting material.

```
internship-project-v1/            # git root
├── CLAUDE.md
├── Prompts/                      # sprint plan + task prompts (sprints.txt, prompt_*.txt)
├── requirements.txt              # stale scratch list; NOT the real deps
├── test.ipynb                    # scratch notebook
├── internship_venv/              # local virtualenv (gitignored)
└── research_assistant/           # ← run everything from here
    ├── .env.example              # LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
    ├── requirements.txt          # real dependency list
    ├── backend/
    │   ├── main.py               # FastAPI app
    │   └── llm_researcher.py     # LLM client + report prompt
    ├── frontend/
    │   ├── streamlit_app.py      # Streamlit entry point
    │   ├── app_pages/            # sidebar.py, input_panel.py, results.py
    │   ├── utils/                # api.py, session.py, file_parser.py, demo_generator.py, styles.py
    │   ├── styles/main.css       # all custom CSS
    │   ├── .streamlit/config.toml
    │   └── demo_research_output.txt
    ├── src/                      # Sprint 1 code (see "Current vs planned layout")
    │   ├── ingestion/            # pdf_loader.py, chunker.py
    │   └── retrieval/            # embeddings.py, vector_store.py
    └── tests/
        └── test_ingestion.py
```

## Existing code

### `backend/main.py`: FastAPI app (v0.2.0)
- `POST /research` takes `ResearchRequest{topic: str (3–500 chars)}`, creates a UUID job, schedules `_run_job` through `BackgroundTasks`, and returns `202` with `JobCreatedResponse{job_id, status="pending"}`.
- `GET /research/{job_id}` returns `JobStatusResponse{job_id, status, topic, result, error}`, or `404`.
- `GET /health` returns `{"status": "ok"}`.
- Job store: the module-level dict `jobs: dict[str, dict]`, in memory only, which supports a single process. Status moves `pending → running → completed | failed`.
- `_run_job` calls `run_research_query(topic)`, saves the Markdown result, and saves `str(exc)` on failure.
- CORS is wide open (`allow_origins=["*"]`) so the Streamlit app can call it.
- It imports `from backend.llm_researcher ...`, so the working directory must be `research_assistant/`.

### `backend/llm_researcher.py`: LLM client
- Uses the `openai` SDK against any OpenAI-compatible endpoint (OpenAI, Gemini compat, Groq, Ollama), chosen by env vars.
- Loads `research_assistant/.env` with `python-dotenv` at import time. It reads `LLM_API_KEY` (required), `LLM_BASE_URL` (default OpenAI), and `LLM_MODEL` (default `gpt-4o-mini`) straight from `os.environ`.
- A lazy singleton client is built by `_get_client()` / `_build_client()`.
- `run_research_query(topic) -> str` sends a system prompt that requires the H2 sections `## Overview`, `## Key Concepts`, `## Current Challenges` and `## Summary`, at 500–1200 words. It uses `temperature=0.4` and `max_tokens=2048`.

### `frontend/streamlit_app.py`: Streamlit UI
(There is no `frontend/app.py`. This file is the entry point.)
- Sets the page config, then calls `init_session()`, `load_css()`, `render_sidebar()` and `render_input_panel()`. It calls `render_results()` only while `job_status` is `pending` or `running`.
- `utils/api.py`: `post_research(topic)` and `poll_job(job_id)` call the backend at a hardcoded `API_BASE = "http://localhost:8000"`. `POLL_INTERVAL_S = 2`.
- `app_pages/input_panel.py`: the centered search bar, file attachments with a preview dialog, and the expanding report output. **Demo fallback:** if the backend can't be reached, it shows a canned report from `utils/demo_generator.py` (`demo_research_output.txt`) with a `demo-<ts>` job id.
- `app_pages/results.py`: polls the backend until the job reaches a terminal status, then writes the result into chat history.
- `app_pages/sidebar.py`: chat history, projects and account section, all held in session state only (nothing is saved).
- `utils/session.py`: session-state defaults (`job_*` keys, `chat_histories`, `projects`, `active_*`) plus `add_chat`, `update_chat`, `open_chat`, `reset_job` and `add_project`.
- `utils/file_parser.py`: `FileInfo` NamedTuple and text extraction for uploads (PDF, docx, xlsx, images).
- Imports are relative to `frontend/` (`from utils.api import ...`) because Streamlit puts the script's directory on `sys.path`.
- Styling: all CSS goes in `styles/main.css`, with no inline styles in components. The theme has light and dark modes in `.streamlit/config.toml`.

### `src/ingestion/` and `src/retrieval/`: Sprint 1 pipeline
- `PDFLoader(path).load() -> list[dict]` returns `{"text", "source", "page"}` per page, using `pypdf`.
- `TextChunker(chunk_size=500, chunk_overlap=50)` is a character-based sliding window (1 token ≈ 4 chars) that snaps chunks to sentence ends. `chunk_pages()` and `chunk_text()` return `TextChunk` dataclasses (`text`, `metadata`, `chunk_id = f"{source}_p{page}_c{idx}"`).
- `EmbeddingModel` wraps sentence-transformers (default `all-MiniLM-L6-v2`, 384-dim). Its embeddings are normalized.
- `VectorStore` is a ChromaDB `PersistentClient` (default `./data/chroma_db`, relative to the working directory) with cosine space. It provides `add_documents` (upsert), `similarity_search -> list[SearchResult]`, `count` and `reset`.

### `tests/test_ingestion.py`
Covers the chunker, embeddings, vector store and an end-to-end retrieval test (top hit cosine similarity > 0.6). It adds `research_assistant/` to `sys.path` itself. The first run downloads the embedding model from HuggingFace.

## Running

The virtualenv is `internship_venv/` at the git root. Run all commands from `research_assistant/`:

```bash
cd research_assistant
cp .env.example .env                                   # then set LLM_API_KEY
pip install -r requirements.txt                        # use THIS requirements file
uvicorn backend.main:app --reload --port 8000          # backend
streamlit run frontend/streamlit_app.py --server.port 8501   # frontend
pytest tests/                                          # tests (offline)
pytest tests/ --live                                   # also run @pytest.mark.live tests (real APIs)
```

## Planned backend layout

New modules go under `research_assistant/backend/`:

| Package | Contents | Sprint |
|---|---|---|
| `backend/config.py` | single pydantic-settings `Settings` object loaded from `.env` | — |
| `backend/tools/` | `web_search.py`, arXiv search tool, Python code-execution sandbox | 4 |
| `backend/ingestion/` | PDF loading, chunking | 1 |
| `backend/retrieval/` | embeddings, ChromaDB store, reranker (two-stage retrieval) | 1–2 |
| `backend/training/` | cross-encoder reranker training (PyTorch), QLoRA fine-tuning (HF + SFTTrainer) | 2–3 |
| `backend/agent/` | ReAct loop, conversation memory | 4 |
| `backend/eval/` | guardrails / hallucination checks, BLEU / ROUGE / BERTScore evaluators | 5 |
| `tests/` | one pytest file per module (`tests/test_<module>.py`) | all |

### Current vs planned layout
- Ingestion and retrieval code is in `src/ingestion/` and `src/retrieval/` today, but the target is `backend/ingestion/` and `backend/retrieval/`. When it moves, update the imports in `src/retrieval/vector_store.py` and `tests/test_ingestion.py`. Don't create a second copy alongside `src/`.
- `backend/config.py` exists (`get_settings()` returns a cached pydantic-settings `Settings`). `llm_researcher.py` hasn't moved to it yet and still reads `os.environ` directly.

### `backend/ingestion/`: PDF ingestion (done; supersedes `src/ingestion`)
- `load_pdf` (PyMuPDF) returns `TextBlock(page, section, text)`. Headings are common section names or short larger/bold lines; everything after References is dropped until an Appendix heading.
- `chunk_blocks` uses tiktoken `cl100k_base` windows of 500 tokens with 50 overlap and never crosses a section. It returns `Chunk(chunk_id=sha256(doc_id:index)[:16], doc_id, text, page, section)`.
- `ingest_pdf(path, out_dir='data/chunks')` uses the file's content hash as `doc_id` and writes `<doc_id>.jsonl`. `POST /documents/upload` stores the file as `data/uploads/<doc_id>.pdf`.

### `backend/tools/web_search.py`: academic search (done)
- `Paper` (pydantic) is the shared paper record. `id` carries a source prefix (`arxiv:1706.03762`, `s2:<paperId>`), and `arxiv_id` is the version-less arXiv ID used for dedup.
- `search_arxiv` uses one shared `arxiv.Client` plus a lock, which keeps at least 3 s between requests.
- `search_semantic_scholar` uses httpx. It retries 429, 5xx and network errors with exponential backoff, honoring `Retry-After`, and doesn't retry other 4xx errors.
- `search_papers` runs the sources in parallel and interleaves them by rank. `deduplicate()` matches on normalized title or arXiv ID and copies missing fields onto the kept record. A failing source is logged and skipped; `SearchUnavailableError` is raised only when every source fails.
- `download_pdf` saves to `data/pdfs/<sanitized id>.pdf`. It checks for the `%PDF-` header and returns `None` on failure instead of raising.
- Each source's results are cached as JSON under `search_cache_dir`, keyed by source, normalized query and count, with a TTL. Failures aren't cached.
- Tests inject `httpx.Client(transport=MockTransport(...))` with the keyword-only `client=` argument, and patch `web_search._sleep` and `web_search.get_settings`.
- Route: `GET /search?q=&max_results=&sources=` returns `list[Paper]`, or 503 if every source is down.

### `backend/agent/`: ReAct agent (done)
- `react.py`: `ReActAgent(llm, store, tool_factory, max_steps).chat(session_id, message) -> AgentResult{answer, citations, steps}`. Replies follow `Thought:` / `Action: {"tool", "args"}` / `Final Answer:`. Bad JSON, unknown tools and bad args come back as Observations. After `agent_max_steps` steps a final answer is forced.
- `tools.py`: `rag_search` (via `retrieval/search.py::retrieve`, which `/rag/search` also uses), `paper_search` (with optional ingest), and `python_interpreter` (`run_python`; not a security sandbox).
- `memory.py`: `SessionStore` saves `data/sessions/<id>.json`, with turns, a stable `Doc N` number per `doc_id`, and citable sources.
- Route: `POST /agent/chat {session_id, message}`. Tests inject a fake LLM and `RagSearchTool(retrieve_fn=...)`.

## Conventions

- **Python 3.10+.** Use `X | None` and built-in generics (`list[str]`, not `typing.List`). Add `from __future__ import annotations` at the top of modules.
- **Type hints** on every function signature and public attribute.
- **Pydantic models for data passed between modules** (chunks, search results, tool calls, agent steps, eval results). Use them in place of bare dicts, dataclasses or ad-hoc classes. Some existing code doesn't follow this yet: `PDFLoader` returns dicts, `TextChunk` is a dataclass, and `SearchResult` is a plain class.
- **Configuration** comes from `.env` through one settings object in `backend/config.py`. Modules import the settings object and don't call `os.getenv` or `load_dotenv` themselves. Every new variable also goes into `.env.example` with a placeholder value.
- **No hardcoded API keys or secrets**, ever. `.env` is gitignored. Values like the frontend's `API_BASE` should become configurable too.
- **pytest for every module.** Mock LLM and network calls so tests run offline and without keys. Use `tmp_path` for ChromaDB and other file state.
- Write docstrings in the existing style: a module header (`name.py` / dashes / description) and Google-style `Args` / `Returns` / `Raises`.
- Use `logging`, not `print`, in backend code.
- Frontend: CSS goes only in `styles/main.css`, and session-state keys go through `utils/session.py`.

## Gotchas

- The root `requirements.txt` is a stale scratch file (it lists stdlib modules such as `pathlib` and `uuid`). Use `research_assistant/requirements.txt`.
- `.gitignore` ignores `pyproject.toml`, so one won't be committed unless that line is removed.
- The job store is in memory: jobs are lost on restart, and it breaks with more than one uvicorn worker.
- The venv's `chromadb 0.3.23` imports pydantic v1's `BaseSettings` and crashes under pydantic 2, so `tests/test_ingestion.py` can't be collected until chromadb is upgraded. Run `pytest tests/test_web_search.py` on its own until then.
- Semantic Scholar's unauthenticated pool often returns 429 on every attempt. Set `SEMANTIC_SCHOLAR_API_KEY` for reliable results.
- When the backend is down, the frontend quietly falls back to demo output. A report that looks right doesn't prove the backend handled it.

## Working style (save tokens)
- Don't read library source code unless something fails.
- Write only 3–5 essential tests per module, no live tests.
- Run only the new test file, once. Fix failures, don't re-run everything.
- Don't explore unrelated files.
- Final summary: max 5 bullet lines.
