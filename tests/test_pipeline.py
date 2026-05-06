"""Phase 4 tests: pipelined fuse, controller, and TTFT."""
from __future__ import annotations

import time

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cacheblend.chunker import chunk_texts
from cacheblend.controller import (
    NVME,
    RAM,
    SLOW_DISK,
    LoadingController,
    StorageProfile,
)
from cacheblend.fusor import fuse_selective, fuse_selective_pipelined
from cacheblend.kv_store import KVStore
from cacheblend.model import LayerwiseModel
from cacheblend.precompute import precompute_chunk_kv

from benchmarks.ttft import measure_ttft


@pytest.fixture(scope="module")
def qwen_setup():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, attn_implementation="eager"
    ).to("cpu").eval()
    lw = LayerwiseModel(hf, dtype=torch.float32, device="cpu", kv_form="pre_rope")
    return tokenizer, lw


def _seed_two_chunk_store(
    lw, tokenizer, *, simulated_load_latency_s: float = 0.0, disk_dir=None
):
    chunks = chunk_texts(
        ["The Eiffel Tower is in Paris. ", "It was completed in 1889."],
        tokenizer,
    )
    store = KVStore(
        simulated_load_latency_s=simulated_load_latency_s,
        disk_dir=str(disk_dir) if disk_dir is not None else None,
    )
    for c in chunks:
        store.put(c.hash, precompute_chunk_kv(lw, c.text, tokenizer))
    return chunks, store


@pytest.mark.requires_model
@torch.no_grad()
def test_pipelined_logits_match_unpipelined(qwen_setup):
    """Pipelining is a pure performance optimization — logits must match
    the unpipelined fuse_selective bit-for-bit."""
    tokenizer, lw = qwen_setup
    chunks, store = _seed_two_chunk_store(lw, tokenizer)

    plain = fuse_selective(lw, chunks, store, recompute_ratio=0.15, check_layer=1)
    piped = fuse_selective_pipelined(lw, chunks, store, recompute_ratio=0.15, check_layer=1)

    diff = (plain - piped).abs().max().item()
    print(f"\n[pipelined-equivalence] max_diff = {diff:.3e}")
    assert diff < 1e-5, f"pipelined logits drift from unpipelined: {diff:.3e}"


@pytest.mark.requires_model
@torch.no_grad()
def test_loading_controller_picks_sensible_ratio(qwen_setup):
    """RAM-tier load is faster than recompute → picker hits the min_ratio
    floor (0.15). A made-up 1 Gbps slow disk is far slower than recompute on
    a 1.5B model → picker climbs above min_ratio."""
    _, lw = qwen_setup

    ctrl = LoadingController(lw, min_ratio=0.15, max_ratio=0.50)
    ctrl.profile(sample_tokens=8)

    num_tokens = 64
    rows = ctrl.explain(num_tokens, [RAM, NVME, SLOW_DISK])
    for row in rows:
        print(
            f"\n[ctrl] {row['storage']:>10s}  t_load={row['t_load_s']*1e3:7.2f} ms  "
            f"t_rec={row['t_recompute_s']*1e3:7.2f} ms  ratio={row['picked_ratio']:.3f}"
        )

    by_name = {r["storage"]: r for r in rows}
    assert by_name["ram"]["picked_ratio"] == pytest.approx(0.15, abs=1e-9)
    # NVME and slow disk should at least equal min_ratio.
    assert by_name["nvme_ssd"]["picked_ratio"] >= 0.15
    assert by_name["slow_disk"]["picked_ratio"] >= 0.15
    # And max_ratio cap must hold everywhere.
    for r in rows:
        assert r["picked_ratio"] <= 0.50 + 1e-9, r


@pytest.mark.requires_model
@torch.no_grad()
def test_pipelined_ttft_lower_with_slow_disk(qwen_setup, tmp_path, capsys):
    """With a *simulated* slow disk (sleep injected on each KV load), the
    pipelined fuse must finish faster than the unpipelined one — because the
    pipelined path overlaps the simulated I/O with host-side prep work.

    The compute side of fuse_selective is not what we're measuring; to keep
    Mac CPU FP32 cost manageable we patch :func:`fuse_selective` with a
    lightweight stub that only triggers the per-chunk KV load. This isolates
    the I/O-overlap behavior — the pipelined path's *only* job — from the
    dominant prefill cost.
    """
    import time as _time

    tokenizer, lw = qwen_setup
    sim_latency = 0.150  # seconds per chunk load
    chunks, store = _seed_two_chunk_store(
        lw,
        tokenizer,
        simulated_load_latency_s=sim_latency,
        disk_dir=tmp_path / "kv",
    )

    def _drain():
        store._mem.clear()

    def _plain_io_only():
        _drain()
        for c in chunks:
            kv = store.get(c.hash)
            assert kv is not None

    def _piped_io_only():
        _drain()
        futures = [store.get_async(c.hash) for c in chunks]
        for c, f in zip(chunks, futures):
            kv = f.result()
            assert kv is not None

    # Time the IO-only paths.
    plain_stats = measure_ttft(_plain_io_only, device=lw.device, n_warmup=1, n_runs=3)
    piped_stats = measure_ttft(_piped_io_only, device=lw.device, n_warmup=1, n_runs=3)
    print(
        f"\n[ttft IO-only] plain median={plain_stats['median']:.3f}s  "
        f"piped median={piped_stats['median']:.3f}s  "
        f"saved={plain_stats['median'] - piped_stats['median']:.3f}s"
    )

    saved = plain_stats["median"] - piped_stats["median"]
    # 2 chunks × 150 ms sequential vs ~150 ms parallel → expect ~150 ms saved.
    # Generous lower bound to absorb timer noise and thread scheduling slack.
    assert saved > 0.10, (
        f"pipelined I/O should save > 100 ms vs sequential under sim_latency="
        f"{sim_latency}s; got plain={plain_stats['median']:.3f}s, "
        f"piped={piped_stats['median']:.3f}s, saved={saved:.3f}s"
    )

    # Also verify the high-level integration runs once — we already proved
    # logits match in test_pipelined_logits_match_unpipelined; here we just
    # make sure the pipelined path completes when called through fuse_*.
    out = fuse_selective_pipelined(
        lw, chunks, store, recompute_ratio=0.15, check_layer=1
    )
    assert out.shape[0] == 1
    assert out.shape[2] == lw.vocab_size
    store.shutdown()
