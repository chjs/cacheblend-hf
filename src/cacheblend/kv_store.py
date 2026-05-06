"""
KVStore — hash-keyed per-chunk per-layer KV storage.

Phase 2 deliverable. In-memory dict by default; opt-in pickled-file backend for
restarts and Phase 4 pipelining experiments.

Storage convention (must match what ``precompute_chunk_kv`` produces):
- value: ``list[tuple[K, V]]`` of length ``num_layers``
- each ``K`` is **pre-RoPE** with shape ``(B=1, num_kv_heads, S_chunk, head_dim)``
- each ``V`` has the same shape, no rotation applied
- both stored on CPU; the fusor moves them to model.device at fuse time

LRU eviction is best-effort: ``put`` records insertion order, ``evict_lru``
drops the oldest entry past the configured cap.

Phase 4 additions:
- ``get_async(chunk_hash) -> Future[ChunkKV]`` — schedule a load on a worker
  thread and return a future. The fusor's pipelined path uses this to overlap
  disk I/O with the previous layer's compute.
- ``prefetch_chunk(chunk_hash)`` — fire-and-forget warmup; subsequent
  ``get`` will hit the in-memory cache.
- A configurable artificial load latency (``simulated_load_latency_s``) lets
  Mac-CPU tests model slow-disk scenarios deterministically.
"""
from __future__ import annotations

import pickle
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import Tensor

LayerKV = Tuple[Tensor, Tensor]           # (K, V) for one layer
ChunkKV = List[LayerKV]                   # full per-layer list for a chunk


class KVStore:
    def __init__(
        self,
        max_chunks: int = 1024,
        disk_dir: Optional[str] = None,
        simulated_load_latency_s: float = 0.0,
        max_async_workers: int = 2,
    ):
        self._mem: "OrderedDict[str, ChunkKV]" = OrderedDict()
        self._max = max_chunks
        self._disk_dir = Path(disk_dir) if disk_dir else None
        if self._disk_dir is not None:
            self._disk_dir.mkdir(parents=True, exist_ok=True)
        self._sim_latency = float(simulated_load_latency_s)
        self._executor: Optional[ThreadPoolExecutor] = None
        self._max_async_workers = max_async_workers

    # --------------------------------------------------------------- core API

    def has(self, chunk_hash: str) -> bool:
        if chunk_hash in self._mem:
            return True
        if self._disk_dir is not None and self._disk_path(chunk_hash).exists():
            return True
        return False

    def get(self, chunk_hash: str) -> Optional[ChunkKV]:
        if chunk_hash in self._mem:
            self._mem.move_to_end(chunk_hash)
            return self._mem[chunk_hash]
        if self._disk_dir is not None:
            # Simulate disk latency on the actual I/O path so sync ``get``
            # and async ``get_async`` pay the same cost — keeps plain vs
            # pipelined comparisons honest.
            if self._sim_latency > 0:
                time.sleep(self._sim_latency)
            kv = self._load_from_disk(chunk_hash)
            if kv is not None:
                self._mem[chunk_hash] = kv
                self._enforce_cap()
            return kv
        return None

    def put(self, chunk_hash: str, kv: ChunkKV) -> None:
        kv = [(k.detach().cpu(), v.detach().cpu()) for k, v in kv]
        self._mem[chunk_hash] = kv
        self._mem.move_to_end(chunk_hash)
        if self._disk_dir is not None:
            self._save_to_disk(chunk_hash, kv)
        self._enforce_cap()

    def evict_lru(self) -> Optional[str]:
        if not self._mem:
            return None
        oldest, _ = self._mem.popitem(last=False)
        return oldest

    def __len__(self) -> int:
        return len(self._mem)

    # ------------------------------------------------------------ async API

    def get_async(self, chunk_hash: str) -> "Future[Optional[ChunkKV]]":
        """Schedule a ``get`` on a worker thread; return a future.

        If the chunk is already in memory, the future resolves immediately
        (still returned as a ``Future`` so callers don't have to special-case).
        For disk hits the worker pays the I/O + simulated latency cost off
        the main thread, freeing it to drive the next layer's compute.
        """
        executor = self._ensure_executor()
        if chunk_hash in self._mem:
            f: Future = Future()
            f.set_result(self._mem[chunk_hash])
            self._mem.move_to_end(chunk_hash)
            return f
        return executor.submit(self._load_blocking, chunk_hash)

    def prefetch_chunk(self, chunk_hash: str) -> "Future[Optional[ChunkKV]]":
        """Fire-and-forget warmup. Returns the future for the caller to track
        if needed; subsequent ``get`` will hit the in-memory cache once the
        future resolves."""
        return self.get_async(chunk_hash)

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def _ensure_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self._max_async_workers)
        return self._executor

    def _load_blocking(self, chunk_hash: str) -> Optional[ChunkKV]:
        # ``get`` already pays the simulated disk latency on cache miss; we
        # don't double-sleep here.
        return self.get(chunk_hash)

    # ---------------------------------------------------------------- helpers

    def _enforce_cap(self) -> None:
        while len(self._mem) > self._max:
            self.evict_lru()

    def _disk_path(self, chunk_hash: str) -> Path:
        assert self._disk_dir is not None
        return self._disk_dir / f"{chunk_hash}.pkl"

    def _save_to_disk(self, chunk_hash: str, kv: ChunkKV) -> None:
        with self._disk_path(chunk_hash).open("wb") as f:
            pickle.dump(kv, f)

    def _load_from_disk(self, chunk_hash: str) -> Optional[ChunkKV]:
        path = self._disk_path(chunk_hash)
        if not path.exists():
            return None
        with path.open("rb") as f:
            return pickle.load(f)
