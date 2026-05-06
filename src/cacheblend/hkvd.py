"""
HKVD — High KV Deviation token selection.

Phase 3 deliverable. The paper's "Insight 1" says: the tokens whose KV would
change most under cross-attention are exactly the tokens worth recomputing,
and recomputing only those (~15%) closes most of the quality gap left by full
reuse. This module owns the math: per-token deviation between cached and
freshly-computed K (post-RoPE), top-k selection, and a placeholder schedule
for per-layer ratio decay.

Why squared L2 over (heads, head_dim)? It matches LMCache's
``LMCBlender.process_qkv`` (``diff_k = sum((k - old_k)**2, dim=hidden)``) —
the simplest order-statistic. Insight 1's plot in the paper uses absolute
attention-deviation contribution, but in practice K-deviation produces the
same ranking and is much cheaper.
"""
from __future__ import annotations

from typing import List

import torch
from torch import Tensor


def kv_deviation(k_fresh: Tensor, k_cached: Tensor) -> Tensor:
    """Per-token squared-L2 deviation between two same-shape K tensors.

    Both inputs must be either pre-RoPE *or* post-RoPE — they should agree.
    Compare post-RoPE if you want the deviation to reflect what attention
    actually sees; pre-RoPE works too and gives a similar ranking.

    Args:
      k_fresh:  (B, num_kv_heads, S, head_dim)
      k_cached: same shape

    Returns:
      (S,) — per-token deviation (B is squeezed; assumes batch size 1).
    """
    if k_fresh.shape != k_cached.shape:
        raise ValueError(
            f"shape mismatch: fresh {tuple(k_fresh.shape)} vs cached {tuple(k_cached.shape)}"
        )
    if k_fresh.dim() != 4:
        raise ValueError(f"expected 4D K (B, H, S, D), got {k_fresh.dim()}D")
    diff = k_fresh.float() - k_cached.float()
    # Sum over heads (dim 1) and head_dim (dim 3). Squeeze batch.
    per_token = (diff * diff).sum(dim=(1, 3))[0]
    return per_token


def select_top_k(deviation: Tensor, k: int) -> Tensor:
    """Indices of the top-``k`` deviations, returned in **ascending order**.

    Sorting matters for downstream causal masking: indices are used as a
    fancy-index into a sequence-shaped tensor and we want positional order.
    """
    S = int(deviation.shape[0])
    if k <= 0 or S == 0:
        return torch.empty(0, dtype=torch.long, device=deviation.device)
    if k >= S:
        return torch.arange(S, device=deviation.device, dtype=torch.long)
    top = torch.topk(deviation, k=k).indices
    return torch.sort(top).values


def gradual_ratio_schedule(
    num_layers: int,
    target_ratio: float = 0.15,
    start_bonus: float = 0.0,
) -> List[float]:
    """Per-layer recompute ratio. v1 returns ``[target_ratio] * num_layers``.

    The signature accommodates a future linear-decay schedule
    (``r_1 = target + start_bonus``, decay to ``target`` over the first half,
    flat thereafter); v1 keeps the schedule flat to mirror LMCache's actual
    behavior and to keep Phase 3's variables tractable. Phase 5 may revisit.
    """
    return [float(target_ratio)] * num_layers
