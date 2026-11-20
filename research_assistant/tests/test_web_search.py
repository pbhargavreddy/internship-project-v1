"""
test_web_search.py
------------------
Tests for backend/tools/web_search.py and the GET /search route.

All unit tests run offline: arXiv is faked by replacing the shared client's
``results`` method, Semantic Scholar and PDF downloads use
``httpx.MockTransport``. The single ``live`` test is skipped unless pytest is
run with ``--live``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import arxiv
import httpx
import pytest
import requests

from backend.config import Settings
from backend.tools import web_search as ws
from backend.tools.web_search import (
    Paper,
    SearchSourceError,
    SearchUnavailableError,
    deduplicate,
    download_pdf,
    normalize_arxiv_id,
    normalize_title,
    search_arxiv,
    search_papers,
    search_semantic_scholar,
)

# ===========================================================================
# Fixtures & helpers
# ===========================================================================


@pytest.fixture(autouse=True)
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Isolated settings: temp cache dir, no API key, no real .env."""
    s = Settings(
        _env_file=None,
        semantic_scholar_api_key=None,
        search_cache_dir=tmp_path / "cache",
        s2_max_retries=3,
        s2_backoff_base_seconds=1.0,
        s2_backoff_max_seconds=30.0,
        arxiv_delay_seconds=3.0,
    )
    monkeypatch.setattr(ws, "get_settings", lambda: s)
    return s


@pytest.fixture(autouse=True)
def sleeps(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record backoff sleeps instead of actually sleeping (live tests sleep for real)."""
    calls: list[float] = []
    if request.node.get_closest_marker("live") is None:
        monkeypatch.setattr(ws, "_sleep", calls.append)
    return calls


def make_arxiv_result(short_id: str, title: str, year: int = 2017) -> arxiv.Result:
    return arxiv.Result(
        entry_id=f"http://arxiv.org/abs/{short_id}",
        published=datetime(year, 6, 12, tzinfo=timezone.utc),
        title=title,
        authors=[arxiv.Result.Author("Ashish Vaswani"), arxiv.Result.Author("Noam Shazeer")],
        summary="The dominant sequence\n  transduction models...",
        links=[
            arxiv.Result.Link(f"http://arxiv.org/abs/{short_id}", title=None, rel="alternate"),
            arxiv.Result.Link(f"http://arxiv.org/pdf/{short_id}", title="pdf", rel="related"),
        ],
    )


@pytest.fixture()
def fake_arxiv(monkeypatch: pytest.MonkeyPatch) -> Callable[..., list[int]]:
    """Install fake arXiv results; returns a list counting API calls."""

    def install(results: list[arxiv.Result] | Exception) -> list[int]:
        calls: list[int] = []

        def fake_results(search: arxiv.Search, offset: int = 0):
            calls.append(1)
            if isinstance(results, Exception):
                raise results
            return iter(results[: search.max_results])

        monkeypatch.setattr(ws._arxiv_client, "results", fake_results)
        return calls

    return install


S2_ITEM = {
    "paperId": "204e3073870fae3d05bcbc2f6a8e263d9b72e776",
    "title": "Attention is All you Need",
    "authors": [{"authorId": "1", "name": "Ashish Vaswani"}, {"authorId": "2", "name": "Noam Shazeer"}],
    "abstract": None,
    "year": 2017,
    "url": "https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776",
    "openAccessPdf": None,
    "citationCount": 120000,
    "externalIds": {"ArXiv": "1706.03762", "DBLP": "conf/nips/VaswaniSPUJGKP17"},
}

S2_ITEM_OTHER = {
    "paperId": "df2b0e26d0599ce3e70df8a9da02e51594e0e992",
    "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    "authors": [{"authorId": "3", "name": "Jacob Devlin"}],
    "abstract": "We introduce a new language representation model called BERT.",
    "year": 2019,
    "url": "https://www.semanticscholar.org/paper/df2b0e26",
    "openAccessPdf": {"url": "https://aclanthology.org/N19-1423.pdf", "status": "GREEN"},
    "citationCount": 90000,
    "externalIds": {"DOI": "10.18653/v1/N19-1423"},
}


def s2_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def s2_ok(items: list[dict]) -> httpx.Response:
    return httpx.Response(200, json={"total": len(items), "offset": 0, "data": items})


def make_paper(**overrides) -> Paper:
    base = dict(
        id="arxiv:1706.03762",
        title="Attention Is All You Need",
        url="https://arxiv.org/abs/1706.03762",
        source="arxiv",
    )
    return Paper(**{**base, **overrides})


# ===========================================================================
# Normalisation
# ===========================================================================


class TestNormalization:
    @pytest.mark.parametrize(
        "a, b",
        [
            ("Attention Is All You Need", "attention is all you need"),
            ("Attention is All you Need.", "  Attention   is all you need "),
            ("BERT: Pre-training of Deep", "BERT - Pre training of deep"),
            ("Café Résumé", "cafe resume"),
        ],
    )
    def test_equivalent_titles_normalize_equal(self, a: str, b: str) -> None:
        assert normalize_title(a) == normalize_title(b)

    def test_different_titles_stay_different(self) -> None:
        assert normalize_title("Attention Is All You Need") != normalize_title("Attention Is Not All You Need")

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("1706.03762v5", "1706.03762"),
            ("arXiv:1706.03762", "1706.03762"),
            ("http://arxiv.org/abs/1706.03762v2", "1706.03762"),
            ("https://arxiv.org/pdf/1706.03762v1.pdf", "1706.03762"),
            ("quant-ph/0201082v1", "quant-ph/0201082"),
            ("", None),
            (None, None),
        ],
    )
    def test_normalize_arxiv_id(self, raw: str | None, expected: str | None) -> None:
        assert normalize_arxiv_id(raw) == expected


# ===========================================================================
# arXiv parsing
# ===========================================================================


class TestSearchArxiv:
    def test_parses_results_into_papers(self, fake_arxiv) -> None:
        fake_arxiv([make_arxiv_result("1706.03762v5", "Attention Is\n  All You Need")])
        papers = search_arxiv("attention", max_results=5)

        assert len(papers) == 1
        p = papers[0]
        assert p.id == "arxiv:1706.03762"
        assert p.arxiv_id == "1706.03762"
        assert p.title == "Attention Is All You Need"  # whitespace collapsed
        assert p.authors == ["Ashish Vaswani", "Noam Shazeer"]
        assert p.abstract == "The dominant sequence transduction models..."
        assert p.year == 2017
        assert p.url == "https://arxiv.org/abs/1706.03762v5"
        assert p.pdf_url == "https://arxiv.org/pdf/1706.03762v5"
        assert p.source == "arxiv"
        assert p.citation_count is None

    def test_respects_max_results(self, fake_arxiv) -> None:
        fake_arxiv([make_arxiv_result(f"2101.0000{i}", f"Paper {i}") for i in range(5)])
        assert len(search_arxiv("q", max_results=2)) == 2

    def test_api_failure_raises_source_error(self, fake_arxiv) -> None:
        fake_arxiv(arxiv.HTTPError("http://export.arxiv.org/api/query", 3, 503))
        with pytest.raises(SearchSourceError, match="arxiv"):
            search_arxiv("q")

    def test_connection_error_raises_source_error(self, fake_arxiv) -> None:
        fake_arxiv(requests.ConnectionError("connection refused"))
        with pytest.raises(SearchSourceError, match="arXiv API request failed"):
            search_arxiv("q")

    def test_rejects_empty_query(self) -> None:
        with pytest.raises(ValueError):
            search_arxiv("   ")

    def test_shared_client_enforces_rate_limit(self) -> None:
        assert ws._arxiv_client.delay_seconds >= 3.0


# ===========================================================================
# Semantic Scholar parsing & retry
# ===========================================================================


class TestSearchSemanticScholar:
    def test_parses_response_and_sends_params(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return s2_ok([S2_ITEM, S2_ITEM_OTHER, {"paperId": "x", "title": None}])

        papers = search_semantic_scholar("transformers", limit=3, client=s2_client(handler))

        assert len(papers) == 2  # entry without title dropped
        attn, bert = papers
        assert attn.id == "s2:204e3073870fae3d05bcbc2f6a8e263d9b72e776"
        assert attn.source == "semantic_scholar"
        assert attn.abstract == ""  # null abstract → empty string
        assert attn.citation_count == 120000
        assert attn.arxiv_id == "1706.03762"
        assert attn.pdf_url == "https://arxiv.org/pdf/1706.03762"  # derived from arXiv ID
        assert bert.pdf_url == "https://aclanthology.org/N19-1423.pdf"
        assert bert.arxiv_id is None

        req = seen[0]
        assert req.url.path == "/graph/v1/paper/search"
        assert req.url.params["query"] == "transformers"
        assert req.url.params["limit"] == "3"
        assert "citationCount" in req.url.params["fields"]
        assert "x-api-key" not in req.headers

    def test_sends_api_key_when_configured(self, settings: Settings) -> None:
        from pydantic import SecretStr

        settings.semantic_scholar_api_key = SecretStr("test-key")
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return s2_ok([])

        search_semantic_scholar("q", client=s2_client(handler))
        assert seen[0].headers["x-api-key"] == "test-key"

    def test_retries_429_with_exponential_backoff(self, sleeps: list[float]) -> None:
        responses = [httpx.Response(429), httpx.Response(429), s2_ok([S2_ITEM])]

        papers = search_semantic_scholar("q", client=s2_client(lambda r: responses.pop(0)))

        assert len(papers) == 1
        assert sleeps == [1.0, 2.0]

    def test_honours_retry_after_header(self, sleeps: list[float]) -> None:
        responses = [httpx.Response(429, headers={"Retry-After": "7"}), s2_ok([])]
        search_semantic_scholar("q", client=s2_client(lambda r: responses.pop(0)))
        assert sleeps == [7.0]

    def test_gives_up_after_max_retries(self, sleeps: list[float]) -> None:
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(429)

        with pytest.raises(SearchSourceError, match="giving up after 4 attempts.*429"):
            search_semantic_scholar("q", client=s2_client(handler))
        assert len(calls) == 4
        assert sleeps == [1.0, 2.0, 4.0]

    def test_retries_server_errors_and_network_errors(self, sleeps: list[float]) -> None:
        def flaky():
            yield httpx.Response(503)
            raise httpx.ConnectError("boom")

        gen = flaky()
        state = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            state["n"] += 1
            if state["n"] <= 2:
                return next(gen)
            return s2_ok([S2_ITEM])

        assert len(search_semantic_scholar("q", client=s2_client(handler))) == 1
        assert len(sleeps) == 2

    def test_client_error_is_not_retried(self, sleeps: list[float]) -> None:
        with pytest.raises(SearchSourceError, match="HTTP 400"):
            search_semantic_scholar(
                "q", client=s2_client(lambda r: httpx.Response(400, text="bad query"))
            )
        assert sleeps == []


# ===========================================================================
# Deduplication & combined search
# ===========================================================================


class TestDeduplicate:
    def test_dedupes_by_normalized_title_and_merges_fields(self) -> None:
        a = make_paper(id="arxiv:1", arxiv_id=None, pdf_url="https://arxiv.org/pdf/1")
        b = make_paper(
            id="s2:abc",
            title="attention is all you need.",
            source="semantic_scholar",
            citation_count=42,
            abstract="An abstract",
        )
        out = deduplicate([a, b])
        assert len(out) == 1
        assert out[0].id == "arxiv:1"  # first occurrence wins
        assert out[0].citation_count == 42  # enriched from duplicate
        assert out[0].abstract == "An abstract"
        assert out[0].pdf_url == "https://arxiv.org/pdf/1"

    def test_dedupes_by_arxiv_id_even_when_titles_differ(self) -> None:
        a = make_paper(arxiv_id="1706.03762")
        b = make_paper(
            id="s2:abc",
            title="Attention is all you need (NeurIPS version)",
            source="semantic_scholar",
            arxiv_id="1706.03762",
        )
        assert len(deduplicate([a, b])) == 1

    def test_keeps_distinct_papers_in_order(self) -> None:
        a = make_paper(id="arxiv:1", title="Paper One", arxiv_id="1")
        b = make_paper(id="arxiv:2", title="Paper Two", arxiv_id="2")
        assert [p.id for p in deduplicate([a, b])] == ["arxiv:1", "arxiv:2"]


class TestSearchPapers:
    def _s2_returns(self, monkeypatch, result: list[dict] | Exception) -> None:
        def fake(query: str, limit: int = 10, *, client=None) -> list[Paper]:
            if isinstance(result, Exception):
                raise result
            return [p for p in (ws._paper_from_s2(i) for i in result) if p][:limit]

        monkeypatch.setattr(ws, "search_semantic_scholar", fake)

    def test_merges_and_dedupes_across_sources(self, fake_arxiv, monkeypatch) -> None:
        fake_arxiv([
            make_arxiv_result("1706.03762v5", "Attention Is All You Need"),
            make_arxiv_result("2005.14165v4", "Language Models are Few-Shot Learners", 2020),
        ])
        self._s2_returns(monkeypatch, [S2_ITEM, S2_ITEM_OTHER])

        papers = search_papers("transformers", max_results=10)

        titles = [p.title for p in papers]
        assert len(papers) == 3
        assert titles.count("Attention Is All You Need") == 1
        attn = papers[0]
        assert attn.source == "arxiv"
        assert attn.citation_count == 120000  # merged in from Semantic Scholar
        assert {p.source for p in papers} == {"arxiv", "semantic_scholar"}

    def test_truncates_to_max_results_interleaving_sources(self, fake_arxiv, monkeypatch) -> None:
        fake_arxiv([make_arxiv_result(f"2101.0000{i}", f"Arxiv Paper {i}") for i in range(3)])
        self._s2_returns(monkeypatch, [S2_ITEM_OTHER])
        papers = search_papers("q", max_results=2)
        assert [p.source for p in papers] == ["arxiv", "semantic_scholar"]

    def test_one_source_failing_does_not_break_the_other(self, fake_arxiv, monkeypatch, caplog) -> None:
        fake_arxiv([make_arxiv_result("1706.03762v5", "Attention Is All You Need")])
        self._s2_returns(monkeypatch, SearchSourceError("semantic_scholar", "rate limited"))

        papers = search_papers("q")

        assert [p.source for p in papers] == ["arxiv"]
        assert "semantic_scholar unavailable" in caplog.text

    def test_unexpected_exception_is_isolated(self, fake_arxiv, monkeypatch) -> None:
        fake_arxiv(RuntimeError("parser exploded"))  # not an arxiv/requests error
        self._s2_returns(monkeypatch, [S2_ITEM_OTHER])
        assert [p.source for p in search_papers("q")] == ["semantic_scholar"]

    def test_all_sources_failing_raises_with_details(self, fake_arxiv, monkeypatch) -> None:
        fake_arxiv(arxiv.HTTPError("u", 3, 503))
        self._s2_returns(monkeypatch, SearchSourceError("semantic_scholar", "down"))
        with pytest.raises(SearchUnavailableError) as excinfo:
            search_papers("q")
        assert set(excinfo.value.errors) == {"arxiv", "semantic_scholar"}

    def test_single_source_selection(self, fake_arxiv, monkeypatch) -> None:
        calls = fake_arxiv([])
        self._s2_returns(monkeypatch, [S2_ITEM])
        papers = search_papers("q", sources=("semantic_scholar",))
        assert calls == []
        assert [p.source for p in papers] == ["semantic_scholar"]

    def test_rejects_unknown_source(self) -> None:
        with pytest.raises(ValueError, match="unknown sources"):
            search_papers("q", sources=("google",))  # type: ignore[arg-type]


# ===========================================================================
# On-disk cache
# ===========================================================================


class TestCache:
    def test_repeated_query_hits_cache(self, fake_arxiv) -> None:
        calls = fake_arxiv([make_arxiv_result("1706.03762v5", "Attention Is All You Need")])
        first = search_arxiv("Attention", max_results=3)
        second = search_arxiv("  attention ", max_results=3)  # same normalised key
        assert calls == [1]
        assert first == second

    def test_different_query_or_count_misses_cache(self, fake_arxiv) -> None:
        calls = fake_arxiv([make_arxiv_result("1706.03762v5", "Attention Is All You Need")])
        search_arxiv("attention", max_results=3)
        search_arxiv("bert", max_results=3)
        search_arxiv("attention", max_results=4)
        assert len(calls) == 3

    def test_s2_cache_avoids_second_request(self) -> None:
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return s2_ok([S2_ITEM])

        client = s2_client(handler)
        search_semantic_scholar("q", client=client)
        search_semantic_scholar("q", client=client)
        assert calls == [1]

    def test_failures_are_not_cached(self, fake_arxiv) -> None:
        fake_arxiv(arxiv.HTTPError("u", 3, 503))
        with pytest.raises(SearchSourceError):
            search_arxiv("q")
        calls = fake_arxiv([make_arxiv_result("1706.03762v5", "Attention Is All You Need")])
        assert len(search_arxiv("q")) == 1
        assert calls == [1]

    def test_expired_entry_is_refetched(self, fake_arxiv, settings: Settings) -> None:
        calls = fake_arxiv([make_arxiv_result("1706.03762v5", "Attention Is All You Need")])
        search_arxiv("q")
        cache_file = next(Path(settings.search_cache_dir).glob("arxiv_*.json"))
        payload = json.loads(cache_file.read_text())
        payload["created_at"] = time.time() - settings.search_cache_ttl_seconds - 1
        cache_file.write_text(json.dumps(payload))
        search_arxiv("q")
        assert len(calls) == 2

    def test_corrupt_cache_file_is_ignored(self, fake_arxiv, settings: Settings) -> None:
        calls = fake_arxiv([make_arxiv_result("1706.03762v5", "Attention Is All You Need")])
        search_arxiv("q")
        next(Path(settings.search_cache_dir).glob("arxiv_*.json")).write_text("{not json")
        assert len(search_arxiv("q")) == 1
        assert len(calls) == 2

    def test_cache_can_be_disabled(self, fake_arxiv, settings: Settings) -> None:
        settings.search_cache_enabled = False
        calls = fake_arxiv([])
        search_arxiv("q")
        search_arxiv("q")
        assert len(calls) == 2
        assert not Path(settings.search_cache_dir).exists()


# ===========================================================================
# PDF download
# ===========================================================================

PDF_BYTES = b"%PDF-1.4\n%fake pdf body\n%%EOF\n"


class TestDownloadPdf:
    def test_saves_pdf_and_returns_path(self, tmp_path: Path) -> None:
        paper = make_paper(pdf_url="https://example.org/paper.pdf")
        client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=PDF_BYTES)))

        path = download_pdf(paper, dest_dir=str(tmp_path / "pdfs"), client=client)

        assert path is not None
        assert Path(path).name == "arxiv_1706.03762.pdf"
        assert Path(path).read_bytes() == PDF_BYTES
        assert not list((tmp_path / "pdfs").glob("*.part"))

    def test_reuses_existing_file(self, tmp_path: Path) -> None:
        paper = make_paper(pdf_url="https://example.org/paper.pdf")
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(200, content=PDF_BYTES)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        download_pdf(paper, dest_dir=str(tmp_path), client=client)
        download_pdf(paper, dest_dir=str(tmp_path), client=client)
        assert calls == [1]

    def test_returns_none_without_pdf_url(self, tmp_path: Path) -> None:
        assert download_pdf(make_paper(pdf_url=None), dest_dir=str(tmp_path)) is None

    def test_returns_none_for_non_pdf_response(self, tmp_path: Path) -> None:
        client = httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"<html>paywall</html>"))
        )
        paper = make_paper(pdf_url="https://example.org/paper.pdf")
        assert download_pdf(paper, dest_dir=str(tmp_path), client=client) is None
        assert list(tmp_path.iterdir()) == []

    def test_returns_none_on_http_error(self, tmp_path: Path) -> None:
        client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
        paper = make_paper(pdf_url="https://example.org/missing.pdf")
        assert download_pdf(paper, dest_dir=str(tmp_path), client=client) is None
        assert list(tmp_path.iterdir()) == []

    def test_sanitizes_legacy_arxiv_ids_in_filename(self, tmp_path: Path) -> None:
        paper = make_paper(id="arxiv:quant-ph/0201082", pdf_url="https://example.org/x.pdf")
        client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=PDF_BYTES)))
        path = download_pdf(paper, dest_dir=str(tmp_path), client=client)
        assert path is not None and Path(path).parent == tmp_path
        assert Path(path).name == "arxiv_quant-ph_0201082.pdf"

    def test_throttles_arxiv_host(self, tmp_path: Path, sleeps: list[float], monkeypatch) -> None:
        monkeypatch.setattr(ws, "_arxiv_pdf_last", time.monotonic())
        client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=PDF_BYTES)))
        download_pdf(make_paper(pdf_url="https://arxiv.org/pdf/1706.03762"), str(tmp_path), client=client)
        assert len(sleeps) == 1 and 0 < sleeps[0] <= 3.0


# ===========================================================================
# FastAPI route
# ===========================================================================


class TestSearchRoute:
    @pytest.fixture()
    def api(self):
        from fastapi.testclient import TestClient

        from backend import main

        return TestClient(main.app), main

    def test_returns_papers(self, api, monkeypatch) -> None:
        client, main = api
        captured: dict = {}

        def fake_search(q, max_results, sources):
            captured.update(q=q, max_results=max_results, sources=sources)
            return [make_paper(citation_count=5)]

        monkeypatch.setattr(main, "search_papers", fake_search)
        resp = client.get("/search", params={"q": "attention", "max_results": 3})

        assert resp.status_code == 200
        body = resp.json()
        assert body[0]["id"] == "arxiv:1706.03762"
        assert body[0]["citation_count"] == 5
        assert captured == {"q": "attention", "max_results": 3, "sources": ["arxiv", "semantic_scholar"]}

    def test_returns_503_when_all_sources_down(self, api, monkeypatch) -> None:
        client, main = api

        def fake_search(*args, **kwargs):
            raise SearchUnavailableError({"arxiv": "down", "semantic_scholar": "down"})

        monkeypatch.setattr(main, "search_papers", fake_search)
        resp = client.get("/search", params={"q": "x"})
        assert resp.status_code == 503
        assert "All search sources failed" in resp.json()["detail"]

    @pytest.mark.parametrize(
        "params", [{}, {"q": ""}, {"q": "   "}, {"q": "x", "max_results": 0}, {"q": "x", "sources": "google"}]
    )
    def test_validation_errors(self, api, params) -> None:
        client, _ = api
        assert client.get("/search", params=params).status_code == 422

    def test_existing_routes_still_work(self, api) -> None:
        client, _ = api
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/research/does-not-exist").status_code == 404


# ===========================================================================
# Live (skipped unless --live)
# ===========================================================================


@pytest.mark.live
def test_live_search_papers_hits_real_apis() -> None:
    papers = search_papers("attention is all you need transformer", max_results=5)
    assert papers, "expected at least one paper from the live APIs"
    assert all(p.title and p.url for p in papers)
    normalized = [normalize_title(p.title) for p in papers]
    assert len(normalized) == len(set(normalized)), "results should be deduplicated"
