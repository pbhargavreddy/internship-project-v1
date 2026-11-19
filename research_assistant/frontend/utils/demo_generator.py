"""
utils/demo_generator.py
-----------------------
Generates demo research reports by reading from demo_research_output.txt.
Ensures the app is 100% functional and interactive even without an active LLM backend.
"""
from __future__ import annotations

import random
from pathlib import Path

DEMO_FILE = Path(__file__).resolve().parents[1] / "demo_research_output.txt"


def get_demo_report(topic: str) -> str:
    """
    Load demo templates from demo_research_output.txt and return a randomized,
    topic-customized report.
    """
    clean_topic = topic.strip() if topic and topic.strip() else "Autonomous AI Systems"

    if not DEMO_FILE.exists():
        return (
            f"# Research Report: {clean_topic}\n\n"
            f"## Overview\n"
            f"An exploration into {clean_topic} reveals key advancements in modern architectures.\n\n"
            f"## Key Concepts\n"
            f"- Semantic chunking and vector retrieval\n"
            f"- Grounded document synthesis\n\n"
            f"## Current Challenges\n"
            f"- Latency and context fragmentation\n\n"
            f"## Summary\n"
            f"Demonstration research report completed for {clean_topic}."
        )

    content = DEMO_FILE.read_text(encoding="utf-8")
    templates = [t.strip() for t in content.split("---") if t.strip()]

    if not templates:
        return f"# Research Report: {clean_topic}\n\nCompleted demo report for {clean_topic}."

    chosen = random.choice(templates)
    # Strip any template header tag like [TEMPLATE_1]
    lines = [line for line in chosen.splitlines() if not line.startswith("[TEMPLATE_")]
    cleaned_template = "\n".join(lines).strip()

    return cleaned_template.replace("{topic}", clean_topic)

