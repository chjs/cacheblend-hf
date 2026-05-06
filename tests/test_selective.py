"""Phase 3 tests: selective KV recompute."""
from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cacheblend.chunker import chunk_texts
from cacheblend.fusor import fuse_full_reuse, fuse_selective
from cacheblend.kv_store import KVStore
from cacheblend.model import LayerwiseModel
from cacheblend.precompute import precompute_chunk_kv


@pytest.fixture(scope="module")
def qwen_setup():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, attn_implementation="eager"
    ).to("cpu").eval()
    lw = LayerwiseModel(hf_model, dtype=torch.float32, device="cpu", kv_form="pre_rope")
    return tokenizer, lw


def _seed_two_chunk_store(lw, tokenizer):
    chunks = chunk_texts(
        [
            "The Eiffel Tower is in Paris. ",
            "It was completed in 1889.",
        ],
        tokenizer,
    )
    store = KVStore()
    for c in chunks:
        store.put(c.hash, precompute_chunk_kv(lw, c.text, tokenizer))
    full_ids = torch.cat([c.token_ids for c in chunks]).unsqueeze(0)
    return chunks, store, full_ids


@pytest.mark.requires_model
@torch.no_grad()
def test_recompute_ratio_zero_equals_full_reuse(qwen_setup):
    """ratio=0 → no HKVD selection; selective collapses to full reuse."""
    tokenizer, lw = qwen_setup
    chunks, store, _ = _seed_two_chunk_store(lw, tokenizer)

    full_reuse = fuse_full_reuse(lw, chunks, store)
    sel_zero = fuse_selective(lw, chunks, store, recompute_ratio=0.0, check_layer=1)

    diff = (full_reuse - sel_zero).abs().max().item()
    print(f"\n[ratio=0] max_diff vs full_reuse = {diff:.3e}")
    assert diff < 1e-5, f"ratio=0 should equal full reuse: {diff:.3e}"


@pytest.mark.requires_model
@torch.no_grad()
def test_recompute_ratio_one_equals_full_recompute(qwen_setup):
    """ratio=1 → every cached-chunk position gets fresh K, V; selective
    collapses to full recompute."""
    tokenizer, lw = qwen_setup
    chunks, store, full_ids = _seed_two_chunk_store(lw, tokenizer)

    sel_one = fuse_selective(lw, chunks, store, recompute_ratio=1.0, check_layer=1)
    recompute = lw.forward_layerwise(full_ids)

    diff = (sel_one - recompute).abs().max().item()
    print(f"\n[ratio=1] max_diff vs full_recompute = {diff:.3e}")
    assert diff < 1e-5, f"ratio=1 should equal full recompute: {diff:.3e}"


@pytest.mark.requires_model
@torch.no_grad()
def test_selective_better_than_full_reuse(qwen_setup):
    """At ratio=0.15, selective recompute must be measurably closer to full
    recompute than full reuse is.

    Threshold note: the paper reports ≥ 50 % L2 reduction at 15 % recompute
    on 4K-token contexts, where attention is sparse and the top-15 % HKVD
    tokens dominate the attention mass. On our 20-token, 2-chunk synthetic
    input the achievable reduction is much smaller — chunk 2 has only 10
    diverging tokens, attention is approximately uniform over them, and a 3-
    token (15 %) selection caps the recoverable error at ≈ 30 %. We assert
    ≥ 15 % reduction here as evidence the algorithm is working; the
    paper-grade ratio is left to Phase 5's full benchmarks.
    """
    tokenizer, lw = qwen_setup
    chunks, store, full_ids = _seed_two_chunk_store(lw, tokenizer)

    recompute = lw.forward_layerwise(full_ids)
    full_reuse = fuse_full_reuse(lw, chunks, store)
    sel = fuse_selective(lw, chunks, store, recompute_ratio=0.15, check_layer=1)

    l2_reuse = (full_reuse - recompute).norm().item()
    l2_sel = (sel - recompute).norm().item()
    print(f"\n[ratio=0.15] L2(full_reuse)={l2_reuse:.3e}, L2(selective)={l2_sel:.3e}, ratio={l2_sel/l2_reuse:.3f}")
    assert l2_sel < l2_reuse * 0.85, (
        f"selective must beat full reuse by ≥15%: "
        f"reuse L2 = {l2_reuse:.3e}, selective L2 = {l2_sel:.3e} "
        f"(ratio {l2_sel/l2_reuse:.3f})"
    )


@pytest.mark.requires_model
@pytest.mark.parametrize("ratio", [0.05, 0.10, 0.15, 0.20])
@torch.no_grad()
def test_quality_vs_ratio(qwen_setup, ratio):
    """Each non-trivial ratio should produce L2 strictly less than full reuse.
    Strict monotonic decrease across ratios isn't required (gradual narrowing
    isn't implemented), but every ratio in this sweep should clear the bar."""
    tokenizer, lw = qwen_setup
    chunks, store, full_ids = _seed_two_chunk_store(lw, tokenizer)

    recompute = lw.forward_layerwise(full_ids)
    full_reuse = fuse_full_reuse(lw, chunks, store)
    sel = fuse_selective(lw, chunks, store, recompute_ratio=ratio, check_layer=1)

    l2_reuse = (full_reuse - recompute).norm().item()
    l2_sel = (sel - recompute).norm().item()
    print(f"\n[ratio={ratio:.2f}] L2(full_reuse)={l2_reuse:.3e}, L2(selective)={l2_sel:.3e}")
    assert l2_sel < l2_reuse, (
        f"selective at ratio={ratio} should beat full reuse: "
        f"reuse {l2_reuse:.3e} vs selective {l2_sel:.3e}"
    )
