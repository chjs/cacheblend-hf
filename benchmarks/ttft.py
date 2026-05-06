"""
TTFT (Time-To-First-Token) measurement helpers.

Phase 4 deliverable. The harness here is intentionally tiny: time the first
forward pass that produces a logits tensor. Two synchronization branches:
- CUDA: ``torch.cuda.synchronize()`` before each ``time.perf_counter()``.
- CPU / MPS: ``time.perf_counter()`` directly; mac-mps may need a sync but
  for our use cases (CPU FP32 in CI, vast.ai CUDA in Phase 5) the two branches
  cover everything.

The result is a small dict so reports can pull median / p50 / p95 directly.
"""
from __future__ import annotations

import statistics
import time
from typing import Callable, Dict, List

import torch


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_ttft(
    method: Callable[[], object],
    *,
    device: torch.device,
    n_warmup: int = 2,
    n_runs: int = 10,
) -> Dict[str, float]:
    """Run ``method()`` ``n_warmup + n_runs`` times; return wall-clock stats.

    The callable must be self-contained — it should accept no arguments and
    return whatever object the caller wants (we discard it). Wrap your fuse_*
    invocation in a lambda.

    Returns a dict with keys: ``median``, ``p50``, ``p95``, ``min``, ``max``,
    ``mean``, ``stdev``, ``n``, all in seconds.
    """
    for _ in range(max(0, n_warmup)):
        method()
        _sync(device)

    samples: List[float] = []
    for _ in range(max(1, n_runs)):
        _sync(device)
        t0 = time.perf_counter()
        method()
        _sync(device)
        samples.append(time.perf_counter() - t0)

    samples.sort()
    p95_idx = max(0, int(round(0.95 * (len(samples) - 1))))
    return {
        "n": float(len(samples)),
        "min": samples[0],
        "max": samples[-1],
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "p50": statistics.median(samples),
        "p95": samples[p95_idx],
        "stdev": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
    }
