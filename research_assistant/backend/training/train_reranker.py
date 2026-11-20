"""
train_reranker.py
-----------------
Train the cross-encoder reranker on MS MARCO (query, positive, negative)
triplets with a margin ranking loss and an explicit PyTorch loop.

Data: ``sentence-transformers/msmarco-bm25`` (config ``triplet``, BM25 hard
negatives). Only the first ``n_train + n_eval`` rows are streamed from the
parquet file via HTTP range reads, so the full 230 MB file isn't downloaded.

Run from ``research_assistant/``::

    python -m backend.training.train_reranker              # 5,000 train / 500 eval
    python -m backend.training.train_reranker --n-train 200 --n-eval 50   # smoke run
    python -m backend.training.train_reranker --limit 2000 --max-length 256 --threads 4
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import pyarrow.parquet as pq
import torch
from huggingface_hub import HfFileSystem
from pydantic import BaseModel
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase, get_linear_schedule_with_warmup

from backend.config import get_settings
from backend.retrieval.reranker import CrossEncoder

log = logging.getLogger(__name__)

DATASET_FILE = "datasets/sentence-transformers/msmarco-bm25/triplet/train-00000-of-00001.parquet"


class Triplet(BaseModel):
    """One training example: a query with a relevant and an irrelevant passage."""

    query: str
    positive: str
    negative: str


def load_triplets(n: int) -> list[Triplet]:
    """Stream the first ``n`` triplets from the Hugging Face parquet file.

    Args:
        n: Number of triplets to read.

    Returns:
        Triplets in file order.
    """
    rows: list[Triplet] = []
    with HfFileSystem().open(DATASET_FILE) as fh:
        for batch in pq.ParquetFile(fh).iter_batches(batch_size=1000):
            rows.extend(Triplet(**r) for r in batch.to_pylist())
            if len(rows) >= n:
                break
    return rows[:n]


class TripletDataset(Dataset):
    """Torch dataset over :class:`Triplet` objects."""

    def __init__(self, triplets: list[Triplet]) -> None:
        self.triplets: list[Triplet] = triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Triplet:
        return self.triplets[idx]


class TripletCollator:
    """Tokenizes a batch into (query, positive) and (query, negative) pair tensors.

    Args:
        tokenizer: Tokenizer of the cross-encoder.
        max_length: Maximum tokens per pair.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _encode(self, queries: list[str], passages: list[str]) -> dict[str, torch.Tensor]:
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

    def __call__(self, batch: list[Triplet]) -> dict[str, dict[str, torch.Tensor]]:
        queries = [t.query for t in batch]
        return {
            "pos": self._encode(queries, [t.positive for t in batch]),
            "neg": self._encode(queries, [t.negative for t in batch]),
        }


def _to(enc: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def evaluate(model: CrossEncoder, loader: DataLoader, device: torch.device) -> float:
    """Fraction of triplets where score(positive) > score(negative)."""
    model.eval()
    correct = total = 0
    for batch in loader:
        pos = model(**_to(batch["pos"], device))
        neg = model(**_to(batch["neg"], device))
        correct += int((pos > neg).sum())
        total += pos.numel()
    return correct / max(total, 1)


def train(
    n_train: int = 5000,
    n_eval: int = 500,
    batch_size: int = 16,
    epochs: int = 1,
    lr: float = 3e-5,
    warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
    log_every: int = 50,
    checkpoint_every: int = 100,
    max_length: int = 256,
    threads: int = 4,
    out_path: Path | None = None,
    seed: int = 42,
) -> float:
    """Train the reranker and save it.

    Args:
        n_train: Training triplets.
        n_eval: Held-out triplets (taken after the training rows).
        batch_size: Triplets per step.
        epochs: Passes over the training set.
        lr: AdamW learning rate.
        warmup_ratio: Fraction of steps used for linear warmup.
        max_grad_norm: Gradient clipping threshold.
        log_every: Log the running loss every this many steps.
        checkpoint_every: Save a checkpoint to ``out_path`` every this many steps (0 disables).
        max_length: Maximum tokens per (query, passage) pair.
        threads: CPU threads for torch (``torch.set_num_threads``).
        out_path: Checkpoint path. Defaults to ``settings.reranker_path``.
        seed: Random seed.

    Returns:
        Held-out accuracy after training.
    """
    settings = get_settings()
    out_path = out_path or settings.reranker_path
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s, threads: %d, max_length: %d", device, threads, max_length)

    log.info("Loading %d triplets from MS MARCO", n_train + n_eval)
    triplets = load_triplets(n_train + n_eval)
    # Rows are grouped by query, so a tail split keeps eval queries mostly unseen.
    train_set, eval_set = triplets[:n_train], triplets[n_train:]

    model = CrossEncoder(max_length=max_length).to(device)
    collate = TripletCollator(model.tokenizer, model.max_length)
    train_loader = DataLoader(
        TripletDataset(train_set), batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    eval_loader = DataLoader(TripletDataset(eval_set), batch_size=batch_size * 2, collate_fn=collate)

    log.info("Accuracy before training: %.3f", evaluate(model, eval_loader, device))

    loss_fn = nn.MarginRankingLoss(margin=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(warmup_ratio * total_steps), num_training_steps=total_steps
    )

    step, running = 0, 0.0
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            pos = model(**_to(batch["pos"], device))
            neg = model(**_to(batch["neg"], device))
            loss = loss_fn(pos, neg, torch.ones_like(pos))  # target 1: pos should rank higher

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            step += 1
            running += loss.item()
            if step % log_every == 0:
                log.info("epoch %d step %d/%d loss %.4f", epoch + 1, step, total_steps, running / log_every)
                running = 0.0
            if checkpoint_every and step % checkpoint_every == 0:
                model.save(out_path)
                log.info("Checkpoint at step %d saved to %s", step, out_path)

    acc = evaluate(model, eval_loader, device)
    log.info("Accuracy after training: %.3f", acc)
    model.save(out_path)
    log.info("Saved reranker to %s", out_path)
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the cross-encoder reranker.")
    parser.add_argument("--n-train", "--limit", dest="n_train", type=int, default=5000,
                        help="training triplets to load (--limit is an alias)")
    parser.add_argument("--n-eval", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=256, help="max tokens per pair")
    parser.add_argument("--threads", type=int, default=4, help="torch CPU threads")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # silence per-request HF Hub logs
    train(
        n_train=args.n_train,
        n_eval=args.n_eval,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        max_length=args.max_length,
        threads=args.threads,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
