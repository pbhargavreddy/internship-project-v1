"""
pdf_loader.py
-------------
Extract page-numbered, section-labelled text blocks from a PDF with PyMuPDF.

Headings are detected with a simple heuristic: a short line that either
matches a common section name (Abstract, Introduction, ... optionally
numbered) or is set larger than the body font / entirely in bold. Everything
under a References / Bibliography heading is dropped, resuming only at an
Appendix-style heading.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

import pymupdf
from pydantic import BaseModel

log = logging.getLogger(__name__)

_NUMBER_PREFIX = r"^(?:(?:\d+(?:\.\d+)*|[IVXLC]+|[A-Z])[.)]?\s+)?"
_COMMON_HEADINGS = re.compile(
    _NUMBER_PREFIX
    + r"(abstract|introduction|related work|background|preliminaries|methods?|methodology|"
    r"approach|model|experiments?|experimental setup|evaluation|results|discussion|analysis|"
    r"limitations|future work|conclusions?|conclusions? and future work|acknowledge?ments?|"
    r"references|bibliography|appendix\b.*|supplementary material)\s*$",
    re.IGNORECASE,
)
_REFERENCES = re.compile(_NUMBER_PREFIX + r"(references|bibliography)\s*$", re.IGNORECASE)
_RESUME_AFTER_REFERENCES = re.compile(
    _NUMBER_PREFIX + r"(appendix\b.*|supplementary material)\s*$", re.IGNORECASE
)

MAX_HEADING_WORDS = 12
LARGER_FONT_RATIO = 1.15
BOLD_FLAG = 16  # PyMuPDF span flag bit for bold


class TextBlock(BaseModel):
    """Contiguous text from one page belonging to one section."""

    page: int  # 1-based
    section: str | None
    text: str


def _is_bold(span: dict) -> bool:
    return bool(span["flags"] & BOLD_FLAG) or "bold" in span["font"].lower()


def _body_font_size(doc: pymupdf.Document, max_pages: int | None = None) -> float:
    """Most common font size, weighted by character count (first ``max_pages`` pages)."""
    sizes: Counter[float] = Counter()
    for page in doc.pages(0, _page_stop(doc, max_pages)):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    sizes[round(span["size"], 1)] += len(span["text"].strip())
    return sizes.most_common(1)[0][0] if sizes else 0.0


def _is_heading(text: str, spans: list[dict], body_size: float) -> bool:
    if _COMMON_HEADINGS.match(text):
        return True
    words = text.split()
    if not words or len(words) > MAX_HEADING_WORDS or not re.search(r"[A-Za-z]", text):
        return False
    if text[-1] in ".,;:":
        return False
    visible = [s for s in spans if s["text"].strip()]
    larger = body_size > 0 and max(s["size"] for s in visible) >= body_size * LARGER_FONT_RATIO
    all_bold = all(_is_bold(s) for s in visible)
    return larger or all_bold


def _clean(lines: list[str]) -> str:
    text = "\n".join(lines)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join hyphenated line breaks
    return re.sub(r"\s+", " ", text).strip()


def _page_stop(doc: pymupdf.Document, max_pages: int | None) -> int:
    return doc.page_count if max_pages is None else min(max_pages, doc.page_count)


def load_pdf(path: str | Path, max_pages: int | None = None) -> list[TextBlock]:
    """
    Read a PDF into section-labelled text blocks, References removed.

    Args:
        path:      Path to the PDF file.
        max_pages: Only read the first N pages (``None`` = all).

    Returns:
        Blocks in reading order; a new block starts at each page or heading.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError:        If the file is not a readable PDF.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    try:
        doc = pymupdf.open(path)
    except (pymupdf.FileDataError, RuntimeError) as exc:
        raise ValueError(f"Not a readable PDF: {path.name}: {exc}") from exc

    blocks: list[TextBlock] = []
    with doc:
        if not doc.is_pdf:
            raise ValueError(f"Not a PDF: {path.name}")
        body_size = _body_font_size(doc, max_pages)
        section: str | None = None
        in_references = False

        for page_no, page in enumerate(doc.pages(0, _page_stop(doc, max_pages)), start=1):
            lines: list[str] = []

            def flush() -> None:
                text = _clean(lines)
                if text and not in_references:
                    blocks.append(TextBlock(page=page_no, section=section, text=text))
                lines.clear()

            for block in page.get_text("dict", sort=True)["blocks"]:
                for line in block.get("lines", []):
                    spans = line["spans"]
                    text = " ".join(s["text"].strip() for s in spans if s["text"].strip())
                    if not text:
                        continue
                    if _is_heading(text, spans, body_size):
                        if in_references and not _RESUME_AFTER_REFERENCES.match(text):
                            continue  # bold author names etc. inside the bibliography
                        flush()
                        in_references = bool(_REFERENCES.match(text))
                        section = text
                        continue
                    lines.append(text)
            flush()

    log.info("Loaded %s: %d blocks, body font %.1fpt", path.name, len(blocks), body_size)
    return blocks
