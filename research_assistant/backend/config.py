"""
config.py
---------
Central settings object for the ScholarPulse backend.

Values are read from environment variables and from ``research_assistant/.env``
(environment variables win). Modules should call :func:`get_settings` instead
of reading ``os.environ`` directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


class Settings(BaseSettings):
    """Typed application settings loaded from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=_ENV_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM (backend/llm/) ───────────────────────────────────────────────────
    # Only "api" (OpenAI-compatible endpoint below). The local transformers model
    # (backend/llm/local_model.py) is too heavy for the server and is never loaded.
    llm_provider: Literal["api"] = "api"
    llm_api_key: SecretStr | None = None
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_reasoning_effort: str = "low"  # sent only to openai/gpt-oss reasoning models
    # Agent fallback chain: comma-separated "groq:<model>" entries, tried in order.
    agent_models: str = "groq:openai/gpt-oss-120b,groq:qwen/qwen3.8-27b,groq:openai/gpt-oss-20b"
    # Optional OpenAI-compatible endpoint appended to the chain when BASE_URL and MODEL are set.
    fallback_base_url: str = ""
    fallback_api_key: SecretStr | None = None
    fallback_model: str = ""
    local_base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"  # notebooks only, not the backend
    local_adapter_dir: Path = Path("data/models/qlora_adapter")

    # ── Fine-tuned summarizer (agent tool summarize_papers) ──────────────────
    # OpenAI-compatible endpoint serving the QLoRA model; empty = always use the API LLM.
    finetuned_base_url: str = ""
    finetuned_model: str = "scholarpulse"
    finetuned_api_key: SecretStr = SecretStr("dummy")

    # ── MongoDB (backend/storage/): chats + GridFS PDF backups ───────────────
    # Unset or unreachable → chats are saved as local JSON (chats_dir). Never log the URI.
    mongodb_uri: SecretStr | None = None
    mongodb_db: str = "scholarpulse"
    chats_dir: Path = Path("data/chats")

    # ── CPU usage ────────────────────────────────────────────────────────────
    cpu_threads: int = 4  # OMP / MKL / torch thread cap, applied at backend startup

    # ── Academic web search (backend/tools/web_search.py) ────────────────────
    semantic_scholar_api_key: SecretStr | None = None
    http_timeout_seconds: float = 20.0
    arxiv_delay_seconds: float = 3.0
    s2_max_retries: int = 4
    s2_backoff_base_seconds: float = 1.0
    s2_backoff_max_seconds: float = 30.0

    # ── On-disk search cache ─────────────────────────────────────────────────
    search_cache_enabled: bool = True
    search_cache_dir: Path = Path("data/cache/search")
    search_cache_ttl_seconds: int = 24 * 60 * 60

    # ── RAG retrieval (backend/retrieval/vector_store.py) ────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    chroma_dir: Path = Path("data/chroma")
    chroma_collection: str = "chunks"

    # ── Cross-encoder reranker (backend/retrieval/reranker.py) ───────────────
    reranker_model: str = "microsoft/MiniLM-L12-H384-uncased"
    reranker_path: Path = Path("data/models/reranker.pt")
    reranker_max_length: int = 256
    rerank_candidates: int = 10
    # Reciprocal-rank fusion of vector and reranker order (score = 1/(k+vec_rank) + 1/(k+rerank_rank)).
    # The trained reranker alone can bury the best vector hit; 0 = reranker order only.
    rerank_fusion_k: int = 60

    # ── paper_search ingestion (backend/agent/tools.py) ──────────────────────
    ingest_max_papers: int = 2
    ingest_max_pages: int = 15
    embedding_batch_size: int = 16

    # ── ReAct agent (backend/agent/) ─────────────────────────────────────────
    sessions_dir: Path = Path("data/sessions")
    agent_max_steps: int = 8
    agent_history_messages: int = 10  # past chat messages (user + assistant) sent to the agent
    agent_history_max_tokens: int = 3000  # history token cap; the oldest turns are dropped first
    # Token budget per agent request (Groq on_demand: 8000 TPM / 7000 input TPM per model, and the
    # request's prompt + max_tokens must fit in one minute's quota or Groq answers 413).
    agent_step_max_tokens: int = 800  # max_tokens of every ReAct reply, including the final answer
    agent_rewrite_max_tokens: int = 100  # standalone-question rewrite
    agent_obs_max_chars: int = 4000  # any tool result sent back to the model
    rag_obs_max_chars: int = 3000  # rag_search: top chunks only, up to this many chars in total
    agent_request_token_budget: int = 6000  # prompt tokens per request; history trimmed first, then old tool results
    agent_output_dir: Path = Path("data/plots")  # python_interpreter PNGs, served at /plots
    agent_raw_log: Path = Path("data/logs/agent_raw.log")  # raw LLM replies that failed to parse
    python_timeout_seconds: float = 15.0

    # ── Evaluation / guardrails (backend/eval/) ──────────────────────────────
    doc_names_file: Path = Path("data/uploads/names.json")  # doc_id -> uploaded filename
    faithfulness_threshold: float = 0.6
    faithfulness_warn_below: float = 0.7

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _reject_local_provider(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().lower() == "local":
            raise ValueError(
                "LLM_PROVIDER=local is disabled: the backend never loads the transformers model "
                "(too CPU-heavy). Use LLM_PROVIDER=api; for a local model point LLM_BASE_URL at Ollama."
            )
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` instance (created once)."""
    return Settings()
