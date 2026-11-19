"""
llm_researcher.py
-----------------
LLM API client that generates a structured markdown research report for a
given topic.

Configured entirely via environment variables so the same code works with:
  • OpenAI          (LLM_BASE_URL=https://api.openai.com/v1)
  • Gemini compat   (LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai)
  • Groq            (LLM_BASE_URL=https://api.groq.com/openai/v1)
  • Ollama (local)  (LLM_BASE_URL=http://localhost:11434/v1, LLM_API_KEY=ollama)
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Load .env from the research_assistant/ root (two levels up from this file)
_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=False)


# ---------------------------------------------------------------------------
# Client factory  (created once per process)
# ---------------------------------------------------------------------------

def _build_client() -> OpenAI:
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        raise EnvironmentError(
            "LLM_API_KEY is not set. Copy .env.example → .env and fill in your key."
        )
    return OpenAI(api_key=api_key, base_url=base_url)


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = _build_client()
    return _client


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert research assistant.  Your task is to produce a thorough,
    well-structured research report in Markdown format.

    The report MUST contain these exact sections with these exact H2 headings:
    ## Overview
    ## Key Concepts
    ## Current Challenges
    ## Summary

    Guidelines:
    - Use clear, precise academic language.
    - Use bullet points, numbered lists, and sub-headings where helpful.
    - Cite relevant details, statistics, or examples where known.
    - Keep the total report between 500 and 1 200 words.
    - Return ONLY the Markdown report — no preamble, no code fences.
""")

_USER_TEMPLATE = "Write a comprehensive research report on the following topic:\n\n**{topic}**"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_research_query(topic: str) -> str:
    """
    Send a structured research prompt to the configured LLM and return a
    comprehensive Markdown report.

    Args:
        topic: The research topic (free-form string from the user).

    Returns:
        A Markdown string containing the full research report.

    Raises:
        EnvironmentError: If LLM_API_KEY is not configured.
        openai.OpenAIError: On API-level failures (rate limit, auth, etc.).
    """
    if not topic or not topic.strip():
        raise ValueError("topic must be a non-empty string")

    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    client = _get_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_TEMPLATE.format(topic=topic.strip())},
        ],
        temperature=0.4,
        max_tokens=2048,
    )

    report = response.choices[0].message.content or ""
    return report.strip()

