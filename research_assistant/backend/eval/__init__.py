"""Evaluation and guardrails: faithfulness checks, prompt-injection filter, metrics."""

from backend.eval.faithfulness import FaithfulnessResult, check_faithfulness, split_sentences
from backend.eval.guardrails import detect_injection, sanitize_text

__all__ = ["FaithfulnessResult", "check_faithfulness", "detect_injection", "sanitize_text", "split_sentences"]
