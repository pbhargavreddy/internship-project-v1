"""
llm_researcher.py
-----------------
LLM API client that generates a structured markdown research report for a
given topic.

The model comes from ``backend.llm.get_llm()``: an OpenAI-compatible endpoint.
"""

from __future__ import annotations

import textwrap

from backend.llm import get_llm


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
        EnvironmentError: If LLM_PROVIDER=api and LLM_API_KEY is not configured.
        openai.OpenAIError: On API-level failures (rate limit, auth, etc.).
    """
    if not topic or not topic.strip():
        raise ValueError("topic must be a non-empty string")

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _USER_TEMPLATE.format(topic=topic.strip())},
    ]
    return get_llm().generate(messages, max_tokens=2048, temperature=0.4)
