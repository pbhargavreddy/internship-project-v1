"""
reranker.py
-----------
Custom PyTorch cross-encoder for second-stage reranking.

A pretrained encoder reads the pair ``[CLS] query [SEP] passage [SEP]``; the
[CLS] vector goes through dropout and a linear head to give one relevance
score. Trained weights (see ``backend/training/train_reranker.py``) are
loaded from ``settings.reranker_path``.

For serving, :func:`load_reranker` loads the model once, puts it in eval mode
and applies dynamic int8 quantization to every ``nn.Linear`` (CPU only).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from backend.config import get_settings
from backend.ingestion import Chunk
from backend.runtime import timed

log = logging.getLogger(__name__)


class CrossEncoder(nn.Module):
    """Encoder + dropout + linear head scoring (query, passage) pairs.

    Args:
        model_name: HuggingFace encoder ID. Defaults to ``settings.reranker_model``.
        encoder: Prebuilt encoder (skips loading ``model_name``; used in tests).
        tokenizer: Prebuilt tokenizer matching ``encoder``.
        dropout: Dropout probability on the [CLS] vector.
        max_length: Maximum tokens per pair. Defaults to ``settings.reranker_max_length``.
    """

    def __init__(
        self,
        model_name: str | None = None,
        encoder: nn.Module | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        dropout: float = 0.1,
        max_length: int | None = None,
    ) -> None:
        super().__init__()
        settings = get_settings()
        self.model_name: str = model_name or settings.reranker_model
        self.max_length: int = max_length or settings.reranker_max_length
        self.encoder: nn.Module = encoder or AutoModel.from_pretrained(self.model_name)
        self.tokenizer: PreTrainedTokenizerBase = tokenizer or AutoTokenizer.from_pretrained(
            self.model_name
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, 1)

    @property
    def device(self) -> torch.device:
        # Not ``head.weight``: after dynamic quantization that is a method, not a tensor.
        return next(self.encoder.parameters()).device

    def tokenize(self, queries: list[str], passages: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize pairs as ``query [SEP] passage``, truncating the passage only.

        Args:
            queries: One query per pair.
            passages: One passage per pair.

        Returns:
            Padded tensors (``input_ids``, ``attention_mask``, ...) on CPU.
        """
        return dict(
            self.tokenizer(
                queries,
                passages,
                padding=True,
                truncation="only_second",
                max_length=self.max_length,
                return_tensors="pt",
            )
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score a batch of tokenized pairs.

        Returns:
            Tensor of shape ``(batch,)`` with one relevance score per pair.
        """
        out = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        cls = out.last_hidden_state[:, 0]
        return self.head(self.dropout(cls)).squeeze(-1)

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Relevance scores for ``query`` against each passage, all pairs in one batch.

        Candidate lists are small (``settings.rerank_candidates``), so one
        forward pass beats several smaller ones.
        """
        if not passages:
            return []
        self.eval()
        with torch.inference_mode():
            enc = {k: v.to(self.device) for k, v in self.tokenize([query] * len(passages), passages).items()}
            return self(**enc).tolist()

    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Sort chunks by cross-encoder score.

        Args:
            query: The user query.
            chunks: Candidate chunks (e.g. first-stage vector-search hits).
            top_k: Number of chunks to keep.

        Returns:
            ``(chunk, score)`` pairs, highest score first.
        """
        if not chunks:
            return []
        with timed("rerank", pairs=len(chunks)):
            scores = self.score(query, [c.text for c in chunks])
        ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_k]

    def save(self, path: str | Path) -> None:
        """Save weights plus the encoder name to ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_name": self.model_name, "state_dict": self.state_dict()}, path)


def quantize_for_cpu(model: CrossEncoder) -> CrossEncoder:
    """Eval mode + dynamic int8 quantization of every ``nn.Linear`` (CPU inference only).

    Args:
        model: A float model on CPU.

    Returns:
        A quantized copy in eval mode.
    """
    model.eval()
    quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    return quantized.eval()


def load_reranker(path: str | Path | None = None) -> CrossEncoder | None:
    """Load a trained reranker for CPU serving (eval mode, int8 linears).

    Args:
        path: Checkpoint file. Defaults to ``settings.reranker_path``.

    Returns:
        The quantized model, or ``None`` if no checkpoint exists.
    """
    settings = get_settings()
    path = Path(path or settings.reranker_path)
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = CrossEncoder(model_name=ckpt["model_name"], max_length=settings.reranker_max_length)
    model.load_state_dict(ckpt["state_dict"])
    model = quantize_for_cpu(model)
    log.info("Loaded reranker from %s (int8 dynamic quantization, max_length=%d)", path, model.max_length)
    return model


_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder | None:
    """Process-wide trained reranker, or ``None`` if none has been trained yet.

    A missing checkpoint is not cached, so a model trained while the server
    runs is picked up on the next call.
    """
    global _reranker
    if _reranker is None:
        _reranker = load_reranker()
    return _reranker
