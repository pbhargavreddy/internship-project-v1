"""
runtime.py
----------
Process-level CPU limits and stage timing for the backend.

:func:`limit_cpu_threads` must run before torch / numpy are imported, so that
``OMP_NUM_THREADS`` and ``MKL_NUM_THREADS`` take effect; ``backend/main.py``
calls it first thing.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager

log = logging.getLogger(__name__)


def limit_cpu_threads(n: int) -> None:
    """Cap OpenMP, MKL and torch at ``n`` threads.

    Args:
        n: Maximum number of compute threads (at least 1).
    """
    n = max(1, n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # HF tokenizers' own thread pool
    try:
        import torch
    except ImportError:
        log.info("CPU threads capped at %d (torch not installed)", n)
        return
    torch.set_num_threads(n)
    log.info("CPU threads capped at %d (OMP, MKL, torch)", n)


@contextmanager
def timed(stage: str, **details: object) -> Iterator[None]:
    """Log how long the ``with`` block took, e.g. ``timing stage=embed ms=412 n=37``.

    Args:
        stage: Stage name (download, chunk, embed, rerank, llm, ...).
        **details: Extra ``key=value`` pairs for the log line.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        extra = "".join(f" {k}={v}" for k, v in details.items())
        log.info("timing stage=%s ms=%.0f%s", stage, (time.perf_counter() - start) * 1000, extra)
