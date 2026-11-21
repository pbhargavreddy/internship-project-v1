"""
test_reranker.py
----------------
Offline tests for backend/retrieval/reranker.py, using a tiny randomly
initialized BERT and a hand-written vocab instead of the pretrained MiniLM.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizer

from backend.ingestion import Chunk
from backend.retrieval.reranker import CrossEncoder

WORDS = "attention transformer image filter network query passage model the a of".split()


@pytest.fixture
def model(tmp_path: Path) -> CrossEncoder:
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]))
    tokenizer = BertTokenizer(str(vocab))
    config = BertConfig(
        vocab_size=len(WORDS) + 5, hidden_size=32, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=64,
    )
    torch.manual_seed(0)
    return CrossEncoder(model_name="tiny", encoder=BertModel(config), tokenizer=tokenizer, max_length=64)


def test_forward_returns_one_score_per_pair(model: CrossEncoder) -> None:
    enc = model.tokenize(["attention model"] * 3, ["the transformer", "an image filter", "query"])
    assert enc["input_ids"][0, 0] == model.tokenizer.cls_token_id
    assert model(**enc).shape == (3,)


def test_rerank_returns_chunks_in_score_order(model: CrossEncoder) -> None:
    texts = ["attention transformer", "image filter network", "the passage of a model", "query"]
    chunks = [Chunk(chunk_id=f"c{i}", doc_id="d", text=t, page=1, section=None) for i, t in enumerate(texts)]
    expected = dict(zip([c.chunk_id for c in chunks], model.score("attention", texts)))

    ranked = model.rerank("attention", chunks, top_k=3)

    scores = [s for _, s in ranked]
    assert len(ranked) == 3
    assert scores == sorted(scores, reverse=True)
    assert scores == sorted(expected.values(), reverse=True)[:3]
    assert all(expected[c.chunk_id] == pytest.approx(s) for c, s in ranked)


def test_rank_fusion_keeps_a_top_vector_hit_the_reranker_buries() -> None:
    from backend.ingestion import Chunk
    from backend.retrieval.search import fuse_ranks

    c = [Chunk(chunk_id=f"c{i}", doc_id="d", text=str(i), page=1, section=None) for i in range(10)]
    vector = c  # c0 is the best vector hit (e.g. R6's "RESIDE dataset" chunk)
    rerank = c[1:] + c[:1]  # the reranker puts it last
    top5 = [chunk.chunk_id for chunk, _ in fuse_ranks(vector, rerank, 60)[:5]]
    assert "c0" in top5 and top5[0] == "c1"  # agreed-on hits still lead
