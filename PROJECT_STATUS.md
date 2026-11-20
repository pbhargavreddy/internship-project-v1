# ScholarPulse: Project Status

Checked 2026-11-20. Paths are relative to `research_assistant/`.

The repo doesn't number steps 0–8 anywhere (`Prompts/sprints.txt` lists Sprints 1–6). The numbering below follows the order in CLAUDE.md, with steps 7 and 8 being eval/guardrails and the UI.

**Tests:** `pytest -q` → **72 passed, 1 skipped, 0 failed** (25 s; the skip is a `@pytest.mark.live` test).

| Step | Area | Status | Evidence |
|---|---|---|---|
| 0 | Setup & config | Done | `backend/config.py` (`get_settings()`, pydantic-settings); `.env.example` covers every setting. `llm_researcher.py` now goes through `backend.llm.get_llm` and no longer reads `os.environ`. |
| 1 | PDF ingestion | Done | `backend/ingestion/` (`pdf_loader.py`, `chunker.py`, `pipeline.py`); `POST /documents/upload`; `tests/test_pdf_ingestion.py`. A leftover old copy is still in `src/ingestion/`. |
| 2 | Academic web search | Done | `backend/tools/web_search.py` (arXiv + Semantic Scholar, dedup, cache); `GET /search`; `tests/test_web_search.py`. |
| 3 | Embeddings + vector store | Done | `backend/retrieval/vector_store.py` (bge + ChromaDB), `search.py`; `GET /rag/search`; ChromaDB holds 177 chunks; `tests/test_vector_store.py`. |
| 4 | Cross-encoder reranker (Sprint 2) | Done | `data/models/reranker.pt` exists (133 MB, 2026-11-20 17:20); `backend/training/train_reranker.py`, `backend/retrieval/reranker.py`; `tests/test_reranker.py`. |
| 5 | QLoRA fine-tuning (Sprint 3) | Done | `data/models/qlora_adapter/` has `adapter_config.json` and `adapter_model.safetensors` (74 MB), plus the tokenizer files (2026-11-20 11:43); `notebooks/finetune_qlora.ipynb`; `backend/llm/local_model.py` (`LLM_PROVIDER=local`); `tests/test_llm.py`. |
| 6 | ReAct agent (Sprint 4) | Done | `backend/agent/` (`react.py`, `tools.py`, `memory.py`); `POST /agent/chat` at `backend/main.py:298`; `tests/test_agent.py`. |
| 7 | Eval & guardrails (Sprint 5) | Partial | `backend/eval/` exists: `faithfulness.py`, `guardrails.py`, `run_eval.py`; wired into `/agent/chat`; `tests/test_eval.py`. **Gaps:** `data/eval/questions.json` has 5 questions, 3 of them placeholders (target is 10); `run_eval.py` has never been run, so there is no `data/eval/results.json`; BLEU isn't implemented. |
| 8 | UI & serving (Sprint 6) | Partial | `frontend/app.py` (upload, document list, agent chat, citations, faithfulness badge, agent steps, figures); `GET /documents`. **Gaps:** no streaming endpoint and no latency/token telemetry in `backend/main.py`; `app.py` has only been compile-checked, never run in a browser. |

## Notes

- **Personal-details PDF (`994ba1859f5c04ee`):** it has 0 chunks in ChromaDB and isn't listed in `names.json`, so it can't come up in retrieval. Its files are still on disk: `data/uploads/994ba1859f5c04ee.pdf` and `data/chunks/994ba1859f5c04ee.jsonl`.
- CLAUDE.md is out of date in two places: it says `llm_researcher.py` still reads `os.environ`, and it says `test_ingestion.py` can't be collected (that file has been deleted).
