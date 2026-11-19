"""
test_ingestion.py
-----------------
End-to-end tests for the Document Ingestion and Vector Retrieval Pipeline.

Tests cover:
1. TextChunker – basic chunking and overlap behaviour.
2. EmbeddingModel – document and query embedding shapes.
3. VectorStore – add + similarity_search with cosine similarity assertion.
4. Full pipeline – mock PDF text → chunks → index → semantic query.
"""

from __future__ import annotations

import io
import tempfile
import uuid
from pathlib import Path

import pytest
import pypdf
from pypdf import PdfWriter

# ---------------------------------------------------------------------------
# Path setup – make sure the package root is importable when running from the
# research_assistant/ directory via  ``pytest tests/``.
# ---------------------------------------------------------------------------
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.chunker import TextChunker, TextChunk
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore


# ===========================================================================
# Fixtures
# ===========================================================================

SAMPLE_RESEARCH_TEXT = """
Abstract

Transformer-based large language models (LLMs) have revolutionised natural language
processing. Pre-training on massive corpora followed by instruction-tuning enables
models such as GPT-4, Claude, and Gemini to generalise across diverse tasks.

Introduction

Attention mechanisms allow every token to attend to every other token in a sequence,
yielding rich contextual representations. The self-attention operation computes queries,
keys, and values from the input embeddings and aggregates information via softmax-
weighted sums.

Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) combines a dense retrieval step with a generative
decoder. Given a user query, the retriever fetches the top-k relevant passages from an
external corpus. The generator then conditions on both the query and the retrieved
passages to produce a grounded, factual response.

Vector Databases

Dense embeddings produced by sentence-transformers encode semantic similarity so that
similar sentences cluster together in the embedding space. ChromaDB persists these
vectors on disk and supports cosine similarity search at query time.

Conclusion

By integrating vector retrieval with language generation, RAG systems dramatically
reduce hallucinations and improve factual accuracy compared with purely parametric LLMs.
"""


def _create_pdf_bytes(text: str) -> bytes:
    """Create an in-memory PDF containing *text* on a single page."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    # pypdf does not natively write text; use a workaround via reportlab or
    # fall back to storing text in the PDF stream directly for testing.
    # For test purposes we write a minimal valid PDF with the text embedded.
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


@pytest.fixture(scope="module")
def embedding_model() -> EmbeddingModel:
    """Shared EmbeddingModel instance (loads once per module)."""
    return EmbeddingModel()


@pytest.fixture()
def fresh_vector_store(tmp_path, embedding_model) -> VectorStore:
    """VectorStore backed by a temporary directory (isolated per test)."""
    return VectorStore(
        collection_name=f"test_{uuid.uuid4().hex[:8]}",
        persist_dir=tmp_path / "chroma_db",
        embedding_model=embedding_model,
    )


# ===========================================================================
# 1. Chunker tests
# ===========================================================================

class TestTextChunker:
    def test_basic_chunking_returns_chunks(self):
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk_text(SAMPLE_RESEARCH_TEXT)
        assert len(chunks) > 0, "Expected at least one chunk"

    def test_chunk_has_text_and_metadata(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_text(SAMPLE_RESEARCH_TEXT, metadata={"source": "test.pdf", "page": 1})
        for chunk in chunks:
            assert isinstance(chunk, TextChunk)
            assert chunk.text.strip(), "Chunk text must not be empty"
            assert chunk.metadata["source"] == "test.pdf"

    def test_overlap_config_validation(self):
        with pytest.raises(ValueError, match="chunk_overlap must be smaller"):
            TextChunker(chunk_size=100, chunk_overlap=100)

    def test_chunk_pages(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        pages = [
            {"text": SAMPLE_RESEARCH_TEXT, "source": "paper.pdf", "page": 1},
        ]
        chunks = chunker.chunk_pages(pages)
        assert len(chunks) > 0
        assert all(c.metadata["source"] == "paper.pdf" for c in chunks)
        assert all(c.metadata["page"] == 1 for c in chunks)

    def test_chunk_ids_are_unique(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        pages = [
            {"text": SAMPLE_RESEARCH_TEXT, "source": "paper.pdf", "page": 1},
        ]
        chunks = chunker.chunk_pages(pages)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"


# ===========================================================================
# 2. EmbeddingModel tests
# ===========================================================================

class TestEmbeddingModel:
    def test_embed_documents_returns_list_of_vectors(self, embedding_model):
        texts = ["Transformers are powerful.", "RAG improves factual accuracy."]
        embeddings = embedding_model.embed_documents(texts)
        assert len(embeddings) == 2
        assert all(isinstance(v, list) for v in embeddings)
        assert all(len(v) == embedding_model.embedding_dimension for v in embeddings)

    def test_embed_query_returns_vector(self, embedding_model):
        vec = embedding_model.embed_query("What is retrieval-augmented generation?")
        assert isinstance(vec, list)
        assert len(vec) == embedding_model.embedding_dimension

    def test_embed_documents_raises_on_empty(self, embedding_model):
        with pytest.raises(ValueError, match="non-empty"):
            embedding_model.embed_documents([])

    def test_embed_query_raises_on_empty(self, embedding_model):
        with pytest.raises(ValueError, match="empty"):
            embedding_model.embed_query("   ")

    def test_embeddings_are_unit_normalised(self, embedding_model):
        import math
        vec = embedding_model.embed_query("unit norm test")
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-4, f"Expected unit norm, got {norm}"


# ===========================================================================
# 3. VectorStore tests
# ===========================================================================

class TestVectorStore:
    def test_add_and_count(self, fresh_vector_store):
        vs = fresh_vector_store
        vs.add_documents(
            chunks=["Transformers use self-attention.", "RAG fetches relevant passages."],
            metadatas=[{"source": "a.pdf", "page": 1}, {"source": "a.pdf", "page": 2}],
            ids=["id_0", "id_1"],
        )
        assert vs.count() == 2

    def test_add_documents_raises_on_empty(self, fresh_vector_store):
        with pytest.raises(ValueError, match="must not be empty"):
            fresh_vector_store.add_documents([], [], [])

    def test_add_documents_raises_on_length_mismatch(self, fresh_vector_store):
        with pytest.raises(ValueError, match="same length"):
            fresh_vector_store.add_documents(
                chunks=["text"],
                metadatas=[],
                ids=["id_0"],
            )

    def test_similarity_search_raises_when_empty(self, fresh_vector_store):
        with pytest.raises(ValueError, match="empty"):
            fresh_vector_store.similarity_search("query")

    def test_similarity_search_returns_results(self, fresh_vector_store):
        vs = fresh_vector_store
        docs = [
            "Dense embeddings encode semantic meaning.",
            "Neural networks learn representations.",
            "ChromaDB stores vectors on disk.",
        ]
        metas = [{"source": "b.pdf", "page": i + 1} for i in range(len(docs))]
        ids = [f"b_id_{i}" for i in range(len(docs))]
        vs.add_documents(docs, metas, ids)

        results = vs.similarity_search("How are vectors stored?", top_k=2)
        assert len(results) == 2
        assert all(hasattr(r, "cosine_similarity") for r in results)
        assert all(hasattr(r, "text") for r in results)
        assert all(hasattr(r, "metadata") for r in results)

    def test_reset_clears_collection(self, fresh_vector_store):
        vs = fresh_vector_store
        vs.add_documents(["Some text."], [{"source": "c.pdf", "page": 1}], ["c_id_0"])
        assert vs.count() == 1
        vs.reset()
        assert vs.count() == 0


# ===========================================================================
# 4. Full pipeline – end-to-end ingestion and semantic retrieval
# ===========================================================================

class TestFullPipeline:
    """
    Indexes the SAMPLE_RESEARCH_TEXT and verifies that a semantic query
    retrieves a relevant chunk with cosine similarity > 0.6.
    """

    def test_end_to_end_semantic_retrieval(self, fresh_vector_store):
        # --- Step 1: Chunk the sample text ---
        chunker = TextChunker(chunk_size=80, chunk_overlap=15)
        pages = [{"text": SAMPLE_RESEARCH_TEXT, "source": "sample.pdf", "page": 1}]
        chunks: list[TextChunk] = chunker.chunk_pages(pages)
        assert chunks, "Chunking produced no output"

        # --- Step 2: Index into VectorStore ---
        vs = fresh_vector_store
        vs.add_documents(
            chunks=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
            ids=[c.chunk_id for c in chunks],
        )
        assert vs.count() == len(chunks)

        # --- Step 3: Query with a semantic question ---
        query = "How does retrieval-augmented generation work?"
        results = vs.similarity_search(query, top_k=5)
        assert results, "No results returned"

        # --- Step 4: Assert relevance ---
        top_result = results[0]
        similarity = top_result.cosine_similarity

        # The top result must contain semantically relevant keywords
        relevant_keywords = {"retrieval", "rag", "query", "passages", "generator", "retriever"}
        text_lower = top_result.text.lower()
        keyword_hit = any(kw in text_lower for kw in relevant_keywords)

        assert keyword_hit, (
            f"Top result does not appear relevant.\n"
            f"Text: {top_result.text!r}\n"
            f"Similarity: {similarity:.4f}"
        )
        assert similarity > 0.6, (
            f"Expected cosine similarity > 0.6, got {similarity:.4f}\n"
            f"Top result: {top_result.text!r}"
        )

    def test_top_result_metadata_has_source_and_page(self, fresh_vector_store):
        chunker = TextChunker(chunk_size=80, chunk_overlap=15)
        pages = [{"text": SAMPLE_RESEARCH_TEXT, "source": "meta_test.pdf", "page": 3}]
        chunks = chunker.chunk_pages(pages)

        vs = fresh_vector_store
        vs.add_documents(
            chunks=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
            ids=[c.chunk_id for c in chunks],
        )

        results = vs.similarity_search("attention mechanism in transformers", top_k=3)
        top = results[0]
        assert "source" in top.metadata
        assert "page" in top.metadata
        assert top.metadata["source"] == "meta_test.pdf"
        assert top.metadata["page"] == 3

