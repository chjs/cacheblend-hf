"""
LoadingController — picks a recompute ratio so KV-load time is hidden.

Phase 4 deliverable. The paper's §5 describes an offline-profiled controller:
measure ``T_recompute(r)`` (linear in ``r``: ``a·r + b``) and ``T_load`` for
each storage tier, then pick the smallest ``r ∈ [r_min, r_max]`` such that
``T_recompute(r) ≥ T_load``. We implement a simplified, dependency-free
version that profiles just enough to make a decision.

Bounds:
- ``min_ratio = 0.15`` — the paper's quality floor; below this F1 falls outside
  the 0.02 budget.
- ``max_ratio = 0.50`` — Phase 3's ratio sweep showed L2 collapses to ~0 at
  r = 0.5 on our 20-token input (and the algorithm degenerates to full
  recompute by r = 1.0). Capping at 0.5 prevents the controller from picking
  pointless excess recompute when the storage tier is so slow that
  T_recompute(0.5) ≤ T_load.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
from transformers.cache_utils import DynamicCache

from cacheblend.model import LayerwiseModel


@dataclass(frozen=True)
class StorageProfile:
    name: str
    throughput_gbps: float                 # GB/s
    cost_per_gb_per_month: float = 0.0     # informational, not used in pick


# Reasonable defaults the report can reference.
RAM = StorageProfile("ram", throughput_gbps=20.0)
NVME = StorageProfile("nvme_ssd", throughput_gbps=3.0)
SATA_SSD = StorageProfile("sata_ssd", throughput_gbps=0.5)
SLOW_DISK = StorageProfile("slow_disk", throughput_gbps=0.1)


class LoadingController:
    """Picks ``recompute_ratio`` based on (storage tier, sequence length).

    Profiling: ``profile(...)`` runs a single full prefill on a small input
    to measure ``prefill_per_token_s``. From that we estimate
    ``T_recompute(r) ≈ r * prefill_per_token_s * num_tokens``. The controller
    treats the ``r=0`` overhead (RoPE, mask building, hook bookkeeping) as
    negligible compared to the recompute share — fine for sub-second TTFTs.
    """

    def __init__(
        self,
        model: LayerwiseModel,
        min_ratio: float = 0.15,
        max_ratio: float = 0.50,
        kv_bytes_per_token: Optional[int] = None,
    ):
        self.model = model
        self.min_ratio = float(min_ratio)
        self.max_ratio = float(max_ratio)
        # KV cost per token across all layers, for both K and V.
        if kv_bytes_per_token is None:
            kv_bytes_per_token = (
                2  # K + V
                * model.num_layers
                * model.num_kv_heads
                * model.head_dim
                * torch.tensor([], dtype=model.dtype).element_size()
            )
        self.kv_bytes_per_token = int(kv_bytes_per_token)
        self.prefill_per_token_s: Optional[float] = None

    # ----------------------------------------------------------------- profiling

    def profile(self, sample_tokens: int = 8) -> float:
        """One-shot offline profile: time a full prefill on ``sample_tokens``
        tokens and divide by ``sample_tokens``. Result cached on ``self``.
        Returns the per-token wall time in seconds."""
        ids = torch.zeros(1, sample_tokens, dtype=torch.long, device=self.model.device)
        # Warmup once to skip first-call jitter.
        with torch.no_grad():
            self.model.forward_layerwise(ids)
        if torch.cuda.is_available() and self.model.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            self.model.forward_layerwise(ids)
        if torch.cuda.is_available() and self.model.device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        self.prefill_per_token_s = dt / max(1, sample_tokens)
        return self.prefill_per_token_s

    # ------------------------------------------------------------ delay estimates

    def estimate_recompute_delay(self, ratio: float, num_tokens: int) -> float:
        if self.prefill_per_token_s is None:
            raise RuntimeError("profile() must be called before estimate_recompute_delay")
        return float(ratio) * self.prefill_per_token_s * num_tokens

    def estimate_load_delay(self, num_tokens: int, storage: StorageProfile) -> float:
        bytes_total = self.kv_bytes_per_token * num_tokens
        bytes_per_s = storage.throughput_gbps * (1024 ** 3)
        return bytes_total / bytes_per_s

    # ------------------------------------------------------------------- decide

    def pick_recompute_ratio(
        self, num_tokens: int, storage: StorageProfile
    ) -> float:
        """Smallest ``r ∈ [min_ratio, max_ratio]`` such that
        ``T_recompute(r) ≥ T_load(num_tokens, storage)``."""
        if self.prefill_per_token_s is None:
            self.profile()
        t_load = self.estimate_load_delay(num_tokens, storage)
        # T_recompute(r) = r * prefill_per_token_s * num_tokens. Solve for r.
        denom = self.prefill_per_token_s * num_tokens
        if denom <= 0:
            return self.min_ratio
        r_match = t_load / denom
        return max(self.min_ratio, min(self.max_ratio, r_match))

    def explain(
        self, num_tokens: int, storages: Iterable[StorageProfile]
    ) -> List[dict]:
        """Returns a list of ``{name, throughput, t_load_s, ratio}`` rows."""
        if self.prefill_per_token_s is None:
            self.profile()
        rows = []
        for s in storages:
            t_load = self.estimate_load_delay(num_tokens, s)
            r = self.pick_recompute_ratio(num_tokens, s)
            t_rec = self.estimate_recompute_delay(r, num_tokens)
            rows.append(
                {
                    "storage": s.name,
                    "throughput_gbps": s.throughput_gbps,
                    "t_load_s": t_load,
                    "t_recompute_s": t_rec,
                    "picked_ratio": r,
                }
            )
        return rows
