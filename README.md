# ScholarPulse

An autonomous AI research assistant: a FastAPI backend runs a ReAct research agent (paper search, RAG over uploaded PDFs, a Python tool), and a Streamlit frontend shows the answers. See `CLAUDE.md` for the layout and the run commands.

## Model files

Large model files aren't stored in git. Put them at these paths, relative to `research_assistant/`:

| File | Size | Where to get it |
|---|---|---|
| `data/models/reranker.pt` | 127 MB | Download: _TODO: add the Kaggle dataset / Google Drive link_. Or retrain it: `python -m backend.training.train_reranker` |

The QLoRA adapter (`data/models/qlora_adapter/`, about 74 MB) is still tracked in git.

Without `reranker.pt` the app still runs: retrieval skips the reranking stage and uses vector-search order. To keep the file somewhere else, set `RERANKER_PATH` in `.env`.
