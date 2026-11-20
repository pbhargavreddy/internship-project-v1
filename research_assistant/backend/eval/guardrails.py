"""
guardrails.py
-------------
Regex prompt-injection filter.

User messages that match are rejected by the API; retrieved passages
(``rag_search`` observations) have the matching lines stripped instead, so a
poisoned PDF can't smuggle instructions into the agent's prompt.

This is a cheap first line of defence, not a complete one: paraphrased
injections will get through.
"""

from __future__ import annotations

import re

INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.I)
    for p in (
        r"\b(ignore|disregard|forget|override)\b.{0,30}\b(previous|prior|above|earlier|all|your)\b.{0,20}"
        r"\b(instructions?|prompts?|rules|directions)\b",
        r"\b(reveal|show|print|display|repeat|output|leak)\b.{0,20}\b(your|the)\s+(system\s+prompt|instructions)\b",
        r"\byou\s+are\s+now\b",
        r"\bact\s+as\s+(an?\s+)?(unrestricted|jailbroken|DAN)\b",
        r"\b(new|updated)\s+instructions\s*:",
        r"\bdeveloper\s+mode\b",
    )
]


def detect_injection(text: str) -> str | None:
    """Return the first injection phrase found in ``text``, or ``None`` if it looks clean."""
    for pattern in INJECTION_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(0)
    return None


def sanitize_text(text: str) -> str:
    """Drop every line of ``text`` that matches an injection pattern."""
    return "\n".join(line for line in text.splitlines() if detect_injection(line) is None)
