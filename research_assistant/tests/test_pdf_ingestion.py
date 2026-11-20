"""
test_pdf_ingestion.py
---------------------
Tests for backend/ingestion using a small PDF generated with PyMuPDF.
"""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest

from backend.ingestion import TextBlock, chunk_blocks, ingest_pdf, load_pdf
from backend.ingestion.chunker import _encoding, make_chunk_id

BODY = "Transformers rely on self-attention to model long-range dependencies. "


@pytest.fixture()
def sample_pdf(tmp_path: Path) -> Path:
    """Two pages: title, Abstract, bold numbered heading, Conclusion, References."""
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), "A Study of Attention", fontsize=20, fontname="hebo")
    p1.insert_text((72, 110), "Abstract", fontsize=11)
    p1.insert_text((72, 130), "We study attention mechanisms in depth.", fontsize=11)
    p1.insert_text((72, 160), "1 Our Method", fontsize=11, fontname="hebo")
    p1.insert_text((72, 180), "The method uses multi-head attention layers.", fontsize=11)
    p2 = doc.new_page()
    p2.insert_text((72, 72), "The method also scales well.", fontsize=11)
    p2.insert_text((72, 100), "Conclusion", fontsize=11)
    p2.insert_text((72, 120), "Attention works well for sequence modelling.", fontsize=11)
    p2.insert_text((72, 150), "References", fontsize=11)
    p2.insert_text((72, 170), "[1] Vaswani et al. Attention is all you need. 2017.", fontsize=11)
    path = tmp_path / "sample.pdf"
    doc.save(path)
    doc.close()
    return path


def test_loader_detects_sections_pages_and_drops_references(sample_pdf: Path) -> None:
    blocks = load_pdf(sample_pdf)
    by_section = {(b.section, b.page): b.text for b in blocks}

    assert by_section[("Abstract", 1)] == "We study attention mechanisms in depth."
    assert "multi-head attention" in by_section[("1 Our Method", 1)]  # bold heading
    assert by_section[("1 Our Method", 2)] == "The method also scales well."  # section spans pages
    assert "sequence modelling" in by_section[("Conclusion", 2)]
    assert all("Vaswani" not in b.text for b in blocks)
    assert all(b.section != "References" for b in blocks)


def test_chunker_token_windows_overlap_and_ids() -> None:
    enc = _encoding()
    blocks = [
        TextBlock(page=1, section="Intro", text=BODY * 80),
        TextBlock(page=2, section="Intro", text=BODY * 80),
        TextBlock(page=2, section="Results", text="Short results section."),
    ]
    chunks = chunk_blocks(blocks, doc_id="doc1")

    intro = [c for c in chunks if c.section == "Intro"]
    assert len(intro) >= 3
    assert all(len(enc.encode(c.text)) <= 500 for c in chunks)
    tail = enc.decode(enc.encode(intro[0].text)[-50:]).strip()
    assert intro[1].text.startswith(tail[:40])  # 50-token overlap
    assert intro[0].page == 1 and intro[-1].page == 2
    assert chunks[-1].section == "Results" and chunks[-1].text == "Short results section."
    assert [c.chunk_id for c in chunks] == [make_chunk_id("doc1", i) for i in range(len(chunks))]


def test_ingest_writes_jsonl_and_upload_endpoint(sample_pdf: Path, tmp_path: Path, monkeypatch) -> None:
    chunks = ingest_pdf(sample_pdf, out_dir=tmp_path / "chunks")
    lines = (tmp_path / "chunks" / f"{chunks[0].doc_id}.jsonl").read_text().splitlines()
    assert [json.loads(line)["chunk_id"] for line in lines] == [c.chunk_id for c in chunks]

    from fastapi.testclient import TestClient

    from backend import main

    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(main, "CHUNKS_DIR", tmp_path / "api_chunks")
    client = TestClient(main.app)

    resp = client.post(
        "/documents/upload",
        files={"file": ("sample.pdf", sample_pdf.read_bytes(), "application/pdf")},
    )
    assert resp.status_code == 200
    assert resp.json()["chunk_count"] == len(chunks)
    assert resp.json()["doc_id"] == chunks[0].doc_id

    bad = client.post("/documents/upload", files={"file": ("x.pdf", b"not a pdf", "application/pdf")})
    assert bad.status_code == 400
