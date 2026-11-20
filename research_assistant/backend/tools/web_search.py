"""
web_search.py
-------------
Academic paper search across arXiv and Semantic Scholar.

Public API:
  Paper                    pydantic model shared with the ingestion / RAG steps
  search_arxiv()           arXiv API via the ``arxiv`` package (rate-limited)
  search_semantic_scholar() Semantic Scholar Graph API via httpx (429 backoff)
  search_papers()          query both sources, deduplicate, merge
  download_pdf()           save a paper's PDF to disk for later ingestion

Each per-source search is cached on disk as JSON keyed by a hash of
(source, normalised query, result count), so repeated queries do not hit the
APIs. Failed requests are never cached.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import arxiv
import httpx
import requests
from pydantic import BaseModel, Field, ValidationError

from backend.config import get_settings

log = logging.getLogger(__name__)

Source = Literal["arxiv", "semantic_scholar"]
ALL_SOURCES: tuple[Source, ...] = ("arxiv", "semantic_scholar")

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "paperId,title,authors,abstract,year,url,openAccessPdf,citationCount,externalIds"
S2_MAX_LIMIT = 100
ARXIV_MAX_PAGE_SIZE = 100
USER_AGENT = "ScholarPulse/0.1 (academic research assistant)"

# Indirection so tests can replace sleeping without slowing down.
_sleep: Callable[[float], None] = time.sleep


# ---------------------------------------------------------------------------
# Models & errors
# ---------------------------------------------------------------------------

class Paper(BaseModel):
    """A single academic paper returned by any search source."""

    id: str = Field(..., description="Source-prefixed ID, e.g. 'arxiv:1706.03762' or 's2:<paperId>'")
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    year: int | None = None
    url: str
    pdf_url: str | None = None
    source: Source
    citation_count: int | None = None
    arxiv_id: str | None = Field(
        default=None, description="Version-less arXiv ID when known (used for deduplication)"
    )


class SearchSourceError(RuntimeError):
    """A single search source failed (API down, bad response, retries exhausted)."""

    def __init__(self, source: Source, message: str) -> None:
        super().__init__(f"{source}: {message}")
        self.source = source


class SearchUnavailableError(RuntimeError):
    """Every requested search source failed."""

    def __init__(self, errors: dict[str, str]) -> None:
        detail = "; ".join(f"{src}: {msg}" for src, msg in errors.items())
        super().__init__(f"All search sources failed — {detail}")
        self.errors = errors


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

_ARXIV_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_WS_RE = re.compile(r"\s+")


def normalize_title(title: str) -> str:
    """Lower-case, strip accents and punctuation, collapse whitespace."""
    decomposed = unicodedata.normalize("NFKD", title)
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _NON_ALNUM_RE.sub(" ", ascii_only.lower()).strip()


def normalize_arxiv_id(raw: str | None) -> str | None:
    """'arXiv:1706.03762v5' → '1706.03762'. Returns None for empty input."""
    if not raw:
        return None
    value = raw.strip()
    value = value.split("arxiv.org/abs/")[-1].split("arxiv.org/pdf/")[-1]
    value = re.sub(r"^arxiv:", "", value, flags=re.IGNORECASE)
    value = value.removesuffix(".pdf")
    value = _ARXIV_VERSION_RE.sub("", value)
    return value.lower() or None


def _clean_text(text: str | None) -> str:
    return _WS_RE.sub(" ", text or "").strip()


# ---------------------------------------------------------------------------
# On-disk cache
# ---------------------------------------------------------------------------

def _cache_path(source: str, query: str, n: int) -> Path:
    key = json.dumps(
        {"source": source, "query": _WS_RE.sub(" ", query.strip().lower()), "n": n},
        sort_keys=True,
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    return Path(get_settings().search_cache_dir) / f"{source}_{digest}.json"


def _cache_read(path: Path) -> list[Paper] | None:
    settings = get_settings()
    if not settings.search_cache_enabled or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - float(payload["created_at"]) > settings.search_cache_ttl_seconds:
            log.debug("Cache expired: %s", path.name)
            return None
        return [Paper.model_validate(p) for p in payload["papers"]]
    except (OSError, ValueError, KeyError, TypeError, ValidationError) as exc:
        log.warning("Ignoring unreadable cache file %s: %s", path, exc)
        return None


def _cache_write(path: Path, query: str, papers: list[Paper]) -> None:
    if not get_settings().search_cache_enabled:
        return
    payload = {
        "created_at": time.time(),
        "query": query,
        "papers": [p.model_dump(mode="json") for p in papers],
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:  # caching is best-effort
        log.warning("Could not write search cache %s: %s", path, exc)


def _cached_search(
    source: Source, query: str, n: int, fetch: Callable[[], list[Paper]]
) -> list[Paper]:
    path = _cache_path(source, query, n)
    cached = _cache_read(path)
    if cached is not None:
        log.info("%s cache hit for %r (%d papers)", source, query, len(cached))
        return cached
    papers = fetch()
    _cache_write(path, query, papers)
    return papers


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------

# One shared client: it remembers when it last called the API and sleeps to
# keep at least ``delay_seconds`` between requests. The lock serialises
# concurrent callers (e.g. parallel FastAPI requests) so they share that limit.
_arxiv_client = arxiv.Client(
    page_size=ARXIV_MAX_PAGE_SIZE,
    delay_seconds=get_settings().arxiv_delay_seconds,
    num_retries=3,
)
_arxiv_lock = threading.Lock()


def _paper_from_arxiv(result: arxiv.Result) -> Paper:
    arxiv_id = normalize_arxiv_id(result.get_short_id()) or result.entry_id
    pdf_url = result.pdf_url.replace("http://", "https://", 1) if result.pdf_url else None
    return Paper(
        id=f"arxiv:{arxiv_id}",
        title=_clean_text(result.title),
        authors=[a.name for a in result.authors],
        abstract=_clean_text(result.summary),
        year=result.published.year if result.published else None,
        url=result.entry_id.replace("http://", "https://", 1),
        pdf_url=pdf_url,
        source="arxiv",
        arxiv_id=arxiv_id,
    )


def search_arxiv(query: str, max_results: int = 10) -> list[Paper]:
    """
    Search arXiv for *query*.

    Args:
        query:       Free-text query (searches all fields).
        max_results: Maximum number of papers to return.

    Returns:
        Papers in arXiv relevance order.

    Raises:
        ValueError:        If the query is empty or ``max_results`` < 1.
        SearchSourceError: If the arXiv API is unreachable or keeps failing.
    """
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if max_results < 1:
        raise ValueError("max_results must be >= 1")

    def fetch() -> list[Paper]:
        search = arxiv.Search(query=query, max_results=max_results)
        try:
            with _arxiv_lock:
                _arxiv_client.page_size = min(max_results, ARXIV_MAX_PAGE_SIZE)
                results = list(_arxiv_client.results(search))
        except (arxiv.ArxivError, requests.RequestException) as exc:
            raise SearchSourceError("arxiv", f"arXiv API request failed: {exc}") from exc
        papers = [_paper_from_arxiv(r) for r in results if r.title]
        log.info("arxiv returned %d papers for %r", len(papers), query)
        return papers

    return _cached_search("arxiv", query, max_results, fetch)


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------

def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
    """Exponential backoff, honouring a numeric ``Retry-After`` header."""
    settings = get_settings()
    delay = settings.s2_backoff_base_seconds * (2**attempt)
    if response is not None:
        retry_after = response.headers.get("Retry-After", "")
        if retry_after.strip().isdigit():
            delay = max(delay, float(retry_after))
    return min(delay, settings.s2_backoff_max_seconds)


def _paper_from_s2(item: dict[str, Any]) -> Paper | None:
    paper_id = item.get("paperId")
    title = _clean_text(item.get("title"))
    if not paper_id or not title:
        return None
    external = item.get("externalIds") or {}
    arxiv_id = normalize_arxiv_id(external.get("ArXiv"))
    pdf_url = (item.get("openAccessPdf") or {}).get("url") or None
    if pdf_url is None and arxiv_id:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    return Paper(
        id=f"s2:{paper_id}",
        title=title,
        authors=[a["name"] for a in item.get("authors") or [] if a.get("name")],
        abstract=_clean_text(item.get("abstract")),
        year=item.get("year"),
        url=item.get("url") or f"https://www.semanticscholar.org/paper/{paper_id}",
        pdf_url=pdf_url,
        source="semantic_scholar",
        citation_count=item.get("citationCount"),
        arxiv_id=arxiv_id,
    )


def _s2_get(client: httpx.Client, params: dict[str, Any]) -> dict[str, Any]:
    """GET the search endpoint, retrying 429 / 5xx / transport errors with backoff."""
    settings = get_settings()
    headers = {"User-Agent": USER_AGENT}
    key = settings.semantic_scholar_api_key
    if key is not None and key.get_secret_value():
        headers["x-api-key"] = key.get_secret_value()

    last_error = "unknown error"
    for attempt in range(settings.s2_max_retries + 1):
        response: httpx.Response | None = None
        try:
            response = client.get(
                S2_SEARCH_URL, params=params, headers=headers, timeout=settings.http_timeout_seconds
            )
        except httpx.TransportError as exc:
            last_error = f"network error: {exc}"
        else:
            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError as exc:
                    raise SearchSourceError("semantic_scholar", f"invalid JSON: {exc}") from exc
            if response.status_code == 429:
                last_error = "rate limited (HTTP 429)"
            elif response.status_code >= 500:
                last_error = f"server error (HTTP {response.status_code})"
            else:  # 4xx other than 429: retrying will not help
                raise SearchSourceError(
                    "semantic_scholar",
                    f"HTTP {response.status_code}: {response.text[:200]}",
                )

        if attempt < settings.s2_max_retries:
            delay = _retry_delay(response, attempt)
            log.warning(
                "Semantic Scholar %s — retry %d/%d in %.1fs",
                last_error, attempt + 1, settings.s2_max_retries, delay,
            )
            _sleep(delay)

    raise SearchSourceError(
        "semantic_scholar",
        f"giving up after {settings.s2_max_retries + 1} attempts: {last_error}",
    )


def search_semantic_scholar(
    query: str, limit: int = 10, *, client: httpx.Client | None = None
) -> list[Paper]:
    """
    Search the Semantic Scholar Graph API (``/graph/v1/paper/search``).

    Uses ``SEMANTIC_SCHOLAR_API_KEY`` from settings when set; works without
    one at a lower shared rate limit.

    Args:
        query:  Free-text query.
        limit:  Maximum number of papers to return (capped at 100 by the API).
        client: Optional ``httpx.Client`` (injected by tests).

    Returns:
        Papers in Semantic Scholar relevance order.

    Raises:
        ValueError:        If the query is empty or ``limit`` < 1.
        SearchSourceError: On non-retryable errors or when retries are exhausted.
    """
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if limit < 1:
        raise ValueError("limit must be >= 1")
    limit = min(limit, S2_MAX_LIMIT)

    def fetch() -> list[Paper]:
        params = {"query": query, "limit": limit, "fields": S2_FIELDS}
        if client is not None:
            payload = _s2_get(client, params)
        else:
            with httpx.Client(follow_redirects=True) as own_client:
                payload = _s2_get(own_client, params)
        items = payload.get("data") or []
        papers = [p for p in (_paper_from_s2(item) for item in items) if p is not None]
        log.info("semantic_scholar returned %d papers for %r", len(papers), query)
        return papers

    return _cached_search("semantic_scholar", query, limit, fetch)


# ---------------------------------------------------------------------------
# Combined search
# ---------------------------------------------------------------------------

def _merge_into(kept: Paper, dup: Paper) -> Paper:
    """Fill missing fields on *kept* from a duplicate record of the same paper."""
    updates: dict[str, Any] = {}
    for field in ("pdf_url", "citation_count", "year", "arxiv_id"):
        if getattr(kept, field) is None and getattr(dup, field) is not None:
            updates[field] = getattr(dup, field)
    if not kept.abstract and dup.abstract:
        updates["abstract"] = dup.abstract
    if not kept.authors and dup.authors:
        updates["authors"] = dup.authors
    return kept.model_copy(update=updates) if updates else kept


def deduplicate(papers: Iterable[Paper]) -> list[Paper]:
    """
    Drop duplicates by arXiv ID or normalised title, keeping first-seen order.

    The first occurrence is kept and enriched with any fields it lacks
    (e.g. ``citation_count`` from Semantic Scholar, ``pdf_url`` from arXiv).
    """
    merged: list[Paper] = []
    by_key: dict[str, int] = {}
    for paper in papers:
        keys = [f"title:{t}" for t in [normalize_title(paper.title)] if t]
        if paper.arxiv_id:
            keys.append(f"arxiv:{paper.arxiv_id}")
        match = next((by_key[k] for k in keys if k in by_key), None)
        if match is None:
            match = len(merged)
            merged.append(paper)
        else:
            merged[match] = _merge_into(merged[match], paper)
        for key in keys:
            by_key.setdefault(key, match)
    return merged


def _interleave(lists: Sequence[list[Paper]]) -> list[Paper]:
    """Round-robin so every source's top hits survive truncation."""
    out: list[Paper] = []
    for i in range(max((len(lst) for lst in lists), default=0)):
        out.extend(lst[i] for lst in lists if i < len(lst))
    return out


def search_papers(
    query: str,
    max_results: int = 10,
    sources: Sequence[Source] = ALL_SOURCES,
) -> list[Paper]:
    """
    Query several sources in parallel, deduplicate and merge the results.

    A failing source is logged and skipped; the others still return.

    Args:
        query:       Free-text query.
        max_results: Maximum number of merged papers to return.
        sources:     Which sources to query.

    Returns:
        Up to ``max_results`` unique papers, interleaving sources by rank.

    Raises:
        ValueError:             On an empty query, bad ``max_results`` or unknown source.
        SearchUnavailableError: If every requested source failed.
    """
    if not query.strip():
        raise ValueError("query must not be empty")
    if max_results < 1:
        raise ValueError("max_results must be >= 1")
    unknown = set(sources) - set(ALL_SOURCES)
    if unknown:
        raise ValueError(f"unknown sources: {sorted(unknown)}")
    selected = list(dict.fromkeys(sources))  # dedupe, keep order
    if not selected:
        raise ValueError("at least one source is required")

    searchers: dict[Source, Callable[[], list[Paper]]] = {
        "arxiv": lambda: search_arxiv(query, max_results),
        "semantic_scholar": lambda: search_semantic_scholar(query, max_results),
    }

    results: dict[Source, list[Paper]] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(selected)) as pool:
        futures = {src: pool.submit(searchers[src]) for src in selected}
        for src, future in futures.items():
            try:
                results[src] = future.result()
            except SearchSourceError as exc:
                log.error("Search source %s unavailable: %s", src, exc)
                errors[src] = str(exc)
            except Exception as exc:  # noqa: BLE001 — never let one source break the other
                log.exception("Unexpected error from search source %s", src)
                errors[src] = f"unexpected error: {exc}"

    if not results:
        raise SearchUnavailableError(errors)

    merged = deduplicate(_interleave([results[s] for s in selected if s in results]))
    return merged[:max_results]


# ---------------------------------------------------------------------------
# PDF download
# ---------------------------------------------------------------------------

_arxiv_pdf_lock = threading.Lock()
_arxiv_pdf_last: float = 0.0


def _throttle_arxiv_host(url: str) -> None:
    """Keep arXiv's ~1 request / 3 s courtesy limit for PDF downloads too."""
    global _arxiv_pdf_last
    host = urlparse(url).hostname or ""
    if not (host == "arxiv.org" or host.endswith(".arxiv.org")):
        return
    with _arxiv_pdf_lock:
        wait = _arxiv_pdf_last + get_settings().arxiv_delay_seconds - time.monotonic()
        if wait > 0:
            _sleep(wait)
        _arxiv_pdf_last = time.monotonic()


def _pdf_filename(paper: Paper) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", paper.id).strip("_") + ".pdf"


def download_pdf(
    paper: Paper, dest_dir: str = "data/pdfs", *, client: httpx.Client | None = None
) -> str | None:
    """
    Download *paper*'s PDF into *dest_dir*.

    Already-downloaded files are reused. Failures are logged, not raised, so
    a batch of downloads can continue past individual bad links.

    Args:
        paper:    Paper with a ``pdf_url``.
        dest_dir: Target directory (created if missing).
        client:   Optional ``httpx.Client`` (injected by tests).

    Returns:
        Path to the saved PDF, or ``None`` if there is no PDF URL, the download
        failed, or the response was not a PDF.
    """
    if not paper.pdf_url:
        log.info("No PDF URL for %s (%r)", paper.id, paper.title)
        return None

    dest = Path(dest_dir)
    target = dest / _pdf_filename(paper)
    if target.is_file() and target.stat().st_size > 0:
        log.info("PDF already downloaded: %s", target)
        return str(target)

    tmp = target.with_suffix(".pdf.part")
    own_client = client is None
    http = client or httpx.Client(follow_redirects=True)
    try:
        dest.mkdir(parents=True, exist_ok=True)
        _throttle_arxiv_host(paper.pdf_url)
        with http.stream(
            "GET",
            paper.pdf_url,
            headers={"User-Agent": USER_AGENT},
            timeout=get_settings().http_timeout_seconds,
        ) as response:
            response.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in response.iter_bytes():
                    fh.write(chunk)
        with tmp.open("rb") as fh:
            if fh.read(5) != b"%PDF-":
                log.warning("Response for %s is not a PDF: %s", paper.id, paper.pdf_url)
                tmp.unlink(missing_ok=True)
                return None
        tmp.replace(target)
        log.info("Downloaded %s → %s", paper.id, target)
        return str(target)
    except (httpx.HTTPError, OSError) as exc:
        log.error("Failed to download PDF for %s from %s: %s", paper.id, paper.pdf_url, exc)
        tmp.unlink(missing_ok=True)
        return None
    finally:
        if own_client:
            http.close()
