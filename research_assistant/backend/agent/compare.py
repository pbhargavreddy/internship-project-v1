"""
compare.py
----------
``compare_papers`` agent tool: a side-by-side comparison of 2–3 of a chat's
documents.

For each document and aspect it retrieves that document's top chunks (vector +
reranker fusion, filename stripped from the query), makes ONE API-LLM call
(Groq fallback chain, never the fine-tuned model) that must return JSON, and
renders the Markdown table, per-paper summaries and key differences in Python.

The tool is *final*: its output is the turn's answer (no 4,000-char tool-result
cut, no agent final-answer step). Without document names and with more than two
documents in the chat it asks which ones and sets ``session.pending_action``.
Enabled with ``ENABLE_COMPARE=true``.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Annotated, Any, ClassVar

from pydantic import BaseModel, BeforeValidator, Field, ValidationError

from backend.agent.doc_resolver import chat_documents, doc_key, resolve_docs
from backend.agent.memory import Citation, PendingAction, Session, count_tokens
from backend.agent.tools import DocRetrieveFn, Tool, _chat_retrieve, strip_doc_name
from backend.config import get_settings
from backend.eval.guardrails import sanitize_text
from backend.ingestion import Chunk
from backend.llm.router import api_llm_label

log = logging.getLogger(__name__)

TOOL_NAME = "compare_papers"
MIN_DOCS, MAX_DOCS = 2, 3
CHUNKS_PER_ASPECT = 2
CHUNK_CHARS = 400
# Default aspects: (table label, retrieval query).
DEFAULT_ASPECTS: list[tuple[str, str]] = [
    ("Objective", "research objective, problem statement and motivation"),
    ("Method", "proposed method, approach and model architecture"),
    ("Dataset", "datasets, benchmarks and experimental setup"),
    ("Results / metrics", "main results, evaluation metrics and performance numbers"),
    ("Limitations", "limitations, weaknesses and future work"),
]
COMPARE_ERROR = ("Sorry, I couldn't build the comparison: the model returned an invalid table twice. "
                 "Please try again in a moment.")
SYSTEM = ("You compare research papers using ONLY the passages given. Reply with one JSON object and nothing "
          "else: no prose, no code fences.")
RETRY_PROMPT = "That reply was not valid JSON ({error}). Reply again with ONLY the JSON object, nothing else."


def _split_names(value: Any) -> Any:
    """Accept ``"a.pdf, b.pdf"`` / ``"a.pdf and b.pdf"`` as well as a list."""
    if isinstance(value, str):
        parts = re.split(r"\s*(?:,|;|\band\b|\bvs\.?|\bversus\b)\s*", value)
        return [p for p in parts if p.strip()]
    return value


class CompareArgs(BaseModel):
    docs: Annotated[list[str] | None, BeforeValidator(_split_names)] = Field(
        None, description="2-3 filenames; omit to use the chat's two documents")
    aspects: Annotated[list[str] | None, BeforeValidator(_split_names)] = None


class CompareJSON(BaseModel):
    """The shape the comparison LLM call must return."""

    rows: list[dict[str, Any]] = Field(..., min_length=1)
    summaries: dict[str, Any] = Field(default_factory=dict)
    differences: list[Any] = Field(default_factory=list)


def parse_compare_json(reply: str) -> CompareJSON:
    """Parse and validate the model's JSON (code fences and text around the object are ignored).

    Raises:
        ValueError: If there is no valid object of the expected shape.
    """
    start = reply.find("{")
    if start == -1:
        raise ValueError("no JSON object found")
    try:
        obj, _ = json.JSONDecoder().raw_decode(reply[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(exc.msg) from exc
    try:
        data = CompareJSON.model_validate(obj)
    except ValidationError as exc:
        raise ValueError(f"wrong shape: {exc.errors()[0]['msg']}") from exc
    if not all("aspect" in row for row in data.rows):
        raise ValueError('every row needs an "aspect" key')
    return data


def _cell(value: Any) -> str:
    text = " ".join(str(value if value is not None else "").split())
    return text.replace("|", "\\|") or "—"


def _lookup(mapping: dict[str, Any], name: str) -> Any:
    """``mapping[name]``, also matching keys spelled differently ("journal1" for "journal1.pdf")."""
    if name in mapping:
        return mapping[name]
    key = doc_key(name)
    return next((v for k, v in mapping.items() if doc_key(k) == key), None)


def render_markdown(data: CompareJSON, docs: list[str], doc_numbers: dict[str, int]) -> str:
    """The comparison table, summaries and key differences, built from the validated JSON."""
    lines = [f"**Comparison of {', '.join(docs[:-1])} and {docs[-1]}**", "",
             "| Aspect | " + " | ".join(_cell(d) for d in docs) + " |",
             "|" + "---|" * (len(docs) + 1)]
    for row in data.rows:
        cells = [_cell(_lookup(row, d)) for d in docs]
        lines.append(f"| {_cell(row.get('aspect'))} | " + " | ".join(cells) + " |")
    lines += ["", "### Summaries"]
    for d in docs:
        summary = " ".join(str(_lookup(data.summaries, d) or "No summary returned.").split())
        lines.append(f"- **{d}** (Doc {doc_numbers[d]}): {summary}")
    if data.differences:
        lines += ["", "### Key differences", *(f"- {' '.join(str(x).split())}" for x in data.differences)]
    return "\n".join(lines)


class ComparePapersTool(Tool):
    """Compares 2–3 documents of the session's chat aspect by aspect (see module docstring).

    Args:
        session: Conversation whose ``Doc N`` numbering, sources and ``pending_action`` are updated.
        doc_retrieve_fn: ``(query, k, doc_ids) -> [(chunk, score)]``. Defaults to reranked retrieval.
        llm: API chat model (Groq chain). Defaults to :func:`backend.llm.get_llm`.
        docs_fn: ``chat_id -> {filename: doc_id}`` in upload order. Defaults to :func:`chat_documents`.
    """

    name = TOOL_NAME
    description = (
        "Compare 2-3 uploaded papers side by side (objective, method, dataset, results, limitations). "
        "docs = list of filenames (omit if the chat has exactly two). Its table is shown to the user as the answer."
    )
    args_model = CompareArgs
    final: ClassVar[bool] = True

    def __init__(
        self,
        session: Session,
        doc_retrieve_fn: DocRetrieveFn | None = None,
        llm: Any | None = None,
        docs_fn: Callable[[str], dict[str, str]] | None = None,
    ) -> None:
        self.session = session
        self.doc_retrieve_fn: DocRetrieveFn = doc_retrieve_fn or _chat_retrieve(session.session_id)
        self._llm = llm
        self.docs_fn = docs_fn or chat_documents
        self.retrieved: list[tuple[int, Chunk]] = []
        self.final_output: str | None = None  # the answer shown to the user (table or clarification)
        settings = get_settings()
        self.max_tokens: int = settings.compare_max_tokens
        self.context_chars: int = settings.compare_context_chars
        self.token_budget: int = settings.compare_token_budget

    def _get_llm(self) -> Any:
        if self._llm is None:
            from backend.llm import get_llm

            self._llm = get_llm()
        return self._llm

    def _finish(self, text: str) -> str:
        self.final_output = text
        return text

    def _clarify(self, names: list[str], reason: str) -> str:
        self.session.pending_action = PendingAction(tool=TOOL_NAME, missing="docs")
        listing = "\n".join(f"- {n}" for n in names)
        return self._finish(f"{reason} This chat has {len(names)} documents:\n{listing}\n\n"
                            f"Reply with two or three of these names (e.g. \"{names[0]} and {names[1]}\").")

    def run(self, args: CompareArgs) -> str:
        self.final_output, self.model_used = None, None
        available = self.docs_fn(self.session.session_id)
        names = list(available)
        if len(names) < MIN_DOCS:
            return self._finish(f"Comparing needs at least two uploaded documents; this chat has {len(names)}. "
                                "Upload another PDF and ask again.")
        if args.docs:
            picked = list(dict.fromkeys(n for ref in args.docs for n in resolve_docs(ref, "", docs=names)))
            unknown = [ref for ref in args.docs if not resolve_docs(ref, "", docs=names)]
            if unknown:
                return self._clarify(names, f"I couldn't find {', '.join(repr(u) for u in unknown)}.")
        elif len(names) == MIN_DOCS:
            picked = names
        else:
            return self._clarify(names, "Which papers should I compare?")
        if len(picked) < MIN_DOCS:
            return self._clarify(names, "Please name at least two papers to compare.")
        if len(picked) > MAX_DOCS:
            return self._clarify(names, f"I can compare at most {MAX_DOCS} papers at a time.")
        self.session.pending_action = None
        aspects = ([(a.strip(), a.strip()) for a in args.aspects if a.strip()] if args.aspects else None) or DEFAULT_ASPECTS
        return self._finish(self._compare(picked, {n: available[n] for n in picked}, aspects))

    def _passages(self, docs: list[str], doc_ids: dict[str, str],
                  aspects: list[tuple[str, str]]) -> dict[str, list[tuple[str, list[Chunk]]]]:
        """Top chunks per (doc, aspect), each doc searched on its own."""
        out: dict[str, list[tuple[str, list[Chunk]]]] = {}
        for d in docs:
            out[d] = []
            for label, query in aspects:
                hits = self.doc_retrieve_fn(strip_doc_name(query, d), CHUNKS_PER_ASPECT, [doc_ids[d]])
                chunks = [c for c, _ in hits[:CHUNKS_PER_ASPECT]]
                out[d].append((label, chunks))
                for c in chunks:
                    n = self.session.doc_number(c.doc_id)
                    self.session.add_source(Citation(doc=n, title=self.session.title_for(c.doc_id),
                                                     page=c.page, section=c.section))
                    self.retrieved.append((n, c))
        return out

    def _prompt(self, docs: list[str], numbers: dict[str, int], aspects: list[tuple[str, str]],
                passages: dict[str, list[tuple[str, list[Chunk]]]], chunk_chars: int) -> list[dict[str, str]]:
        blocks = []
        for d in docs:
            parts = [f"=== {d} (cite as Doc {numbers[d]}) ==="]
            for label, chunks in passages[d]:
                texts = [f"[Doc {numbers[d]}, Sec {c.section or 'n/a'}] {sanitize_text(c.text).strip()[:chunk_chars]}"
                         for c in chunks] or ["(no passage found)"]
                parts.append(f"-- {label} --\n" + "\n".join(texts))
            blocks.append("\n".join(parts))
        cols = ", ".join(f'"{d}": "<short cell with [Doc {numbers[d]}, Sec x] citation>"' for d in docs)
        schema = ('{"rows": [{"aspect": "<aspect>", ' + cols + '}], '
                  '"summaries": {' + ", ".join(f'"{d}": "<2-3 sentences>"' for d in docs) + '}, '
                  '"differences": ["<one key difference>", "..."]}')
        user = (f"Compare these papers on these aspects, one row each, in this order: "
                f"{', '.join(a for a, _ in aspects)}.\nKeep each cell under 30 words and cite it as [Doc N, Sec S] "
                f"from the passage labels. Write \"not reported\" when the passages don't say.\n"
                f"Give 3-5 key differences.\n\nReturn exactly this JSON shape:\n{schema}\n\n" + "\n\n".join(blocks))
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]

    def _fit_prompt(self, docs: list[str], numbers: dict[str, int], aspects: list[tuple[str, str]],
                    passages: dict[str, list[tuple[str, list[Chunk]]]]) -> list[dict[str, str]]:
        """The prompt with every doc given the same text share, shrunk until prompt + max_tokens fits the budget."""
        per_chunk = min(CHUNK_CHARS, self.context_chars // (len(docs) * len(aspects) * CHUNKS_PER_ASPECT))
        while True:
            messages = self._prompt(docs, numbers, aspects, passages, per_chunk)
            tokens = sum(count_tokens(m["content"]) for m in messages)
            if tokens + self.max_tokens <= self.token_budget or per_chunk <= 60:
                log.info("compare_papers prompt: %d tokens + max_tokens %d (budget %d), %d chars per chunk",
                         tokens, self.max_tokens, self.token_budget, per_chunk)
                return messages
            per_chunk = int(per_chunk * 0.75)

    def _call(self, messages: list[dict[str, str]]) -> str:
        llm = self._get_llm()
        reply = llm.generate(messages, max_tokens=self.max_tokens, temperature=0.0)
        usage = getattr(llm, "last_usage", None)
        log.info("compare_papers LLM call: prompt≈%d tokens, max_tokens=%d, usage=%s",
                 sum(count_tokens(m["content"]) for m in messages), self.max_tokens, usage)
        return reply or ""

    def _compare(self, docs: list[str], doc_ids: dict[str, str], aspects: list[tuple[str, str]]) -> str:
        passages = self._passages(docs, doc_ids, aspects)
        numbers = {d: self.session.doc_number(doc_ids[d]) for d in docs}
        messages = self._fit_prompt(docs, numbers, aspects, passages)
        data: CompareJSON | None = None
        try:
            reply = self._call(messages)
            try:
                data = parse_compare_json(reply)
            except ValueError as exc:
                log.warning("compare_papers: invalid JSON (%s); retrying once", exc)
                retry = [*messages, {"role": "assistant", "content": reply[:2000] or "(empty)"},
                         {"role": "user", "content": RETRY_PROMPT.format(error=exc)}]
                data = parse_compare_json(self._call(retry))
        except ValueError as exc:
            log.warning("compare_papers: invalid JSON again (%s); giving up", exc)
        except Exception as exc:  # noqa: BLE001 — LLM failure: clean message instead of a crash
            log.warning("compare_papers LLM call failed: %s", exc)
            return f"Sorry, I couldn't compare the papers right now ({type(exc).__name__}). Please try again."
        last = getattr(self._llm, "last_reply", None)
        self.model_used = (f"{'Groq ' if last.model.provider == 'groq' else ''}{last.model.short_name}"
                           if last is not None else api_llm_label())
        if data is None:
            return COMPARE_ERROR
        return render_markdown(data, docs, numbers)
