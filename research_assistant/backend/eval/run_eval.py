"""
run_eval.py
-----------
Offline evaluation of the ReAct agent on hand-written questions.

Each question in ``data/eval/questions.json`` (``[{"question", "reference"}]``)
is answered by the agent in a fresh session. Answers are scored against the
references with ROUGE-L and BERTScore F1, and checked for faithfulness
against the chunks they cite. Results go to ``data/eval/results.json``.

Usage (from research_assistant/):
    python -m backend.eval.run_eval [--questions PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from pathlib import Path

import evaluate

from backend.agent import ReActAgent
from backend.eval.faithfulness import check_faithfulness

log = logging.getLogger(__name__)

QUESTIONS_PATH = Path("data/eval/questions.json")
RESULTS_PATH = Path("data/eval/results.json")


def run(questions_path: Path = QUESTIONS_PATH, out_path: Path = RESULTS_PATH) -> dict:
    """Answer every question, score the answers and write the results file.

    Args:
        questions_path: JSON list of ``{"question", "reference"}`` objects.
        out_path: Where the per-question rows and averages are written.

    Returns:
        ``{"rows": [...], "averages": {...}}``.
    """
    items = json.loads(questions_path.read_text(encoding="utf-8"))
    agent = ReActAgent()
    answers, faith = [], []
    for i, item in enumerate(items, 1):
        log.info("Q%d/%d: %s", i, len(items), item["question"])
        result = agent.chat(f"eval-{uuid.uuid4().hex[:8]}", item["question"])
        answers.append(result.answer)
        faith.append(check_faithfulness(result.answer, result.cited_chunks).score if result.cited_chunks else None)

    refs = [item["reference"] for item in items]
    rouge = evaluate.load("rouge").compute(predictions=answers, references=refs, use_aggregator=False)["rougeL"]
    bert = evaluate.load("bertscore").compute(predictions=answers, references=refs, lang="en")["f1"]

    rows = [
        {"question": it["question"], "answer": a, "reference": it["reference"],
         "rougeL": round(r, 4), "bertscore_f1": round(b, 4), "faithfulness": f}
        for it, a, r, b, f in zip(items, answers, rouge, bert, faith)
    ]
    scored = [f for f in faith if f is not None]
    averages = {
        "rougeL": round(sum(rouge) / len(rouge), 4),
        "bertscore_f1": round(sum(bert) / len(bert), 4),
        "faithfulness": round(sum(scored) / len(scored), 4) if scored else None,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"rows": rows, "averages": averages}, indent=2), encoding="utf-8")
    return {"rows": rows, "averages": averages}


def _fmt(v: float | None) -> str:
    return "  n/a" if v is None else f"{v:.3f}"


def print_table(results: dict) -> None:
    """Print one row per question plus the averages."""
    print(f"{'#':>2}  {'ROUGE-L':>7}  {'BERT-F1':>7}  {'Faith':>5}  Question")
    for i, r in enumerate(results["rows"], 1):
        print(f"{i:>2}  {_fmt(r['rougeL']):>7}  {_fmt(r['bertscore_f1']):>7}  {_fmt(r['faithfulness']):>5}  "
              f"{r['question'][:60]}")
    a = results["averages"]
    print(f"{'avg':>3} {_fmt(a['rougeL']):>7}  {_fmt(a['bertscore_f1']):>7}  {_fmt(a['faithfulness']):>5}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the ReAct agent")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print_table(run(args.questions, args.out))
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
