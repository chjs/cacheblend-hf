"""Phase 2 tests: chunk store, RoPE shift, and full-reuse fusion."""
from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from cacheblend.chunker import chunk_texts
from cacheblend.fusor import fuse_full_reuse
from cacheblend.kv_store import KVStore
from cacheblend.model import LayerwiseModel
from cacheblend.precompute import precompute_chunk_kv
from cacheblend.rope import apply_rope_shift


# Reuse the model across all module tests; loading 1.5B FP32 takes ~16s.
@pytest.fixture(scope="module")
def qwen_setup():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, attn_implementation="eager"
    ).to("cpu").eval()
    lw = LayerwiseModel(hf_model, dtype=torch.float32, device="cpu", kv_form="pre_rope")
    return tokenizer, lw


@pytest.mark.requires_model
@torch.no_grad()
def test_rope_shift_correctness(qwen_setup):
    """K rotated to position P via apply_rope_shift must equal HF's
    apply_rotary_pos_emb evaluated at position P on the same pre-RoPE K."""
    _, lw = qwen_setup

    S = 6
    P = 17
    k_pre = torch.randn(1, lw.num_kv_heads, S, lw.head_dim, dtype=lw.dtype)

    # Our shift.
    k_post_ours = apply_rope_shift(k_pre, torch.arange(P, P + S), lw)

    # Reference: HF's own apply_rotary_pos_emb at the same target positions.
    dummy_hidden = torch.zeros(1, S, lw.hidden_size, dtype=lw.dtype)
    pos_ids = torch.arange(P, P + S).unsqueeze(0)
    cos, sin = lw.compute_position_embeddings(dummy_hidden, pos_ids)
    dummy_q = torch.zeros_like(k_pre)
    _, k_post_ref = apply_rotary_pos_emb(dummy_q, k_pre, cos, sin)

    diff = (k_post_ours - k_post_ref).abs().max().item()
    print(f"\n[rope shift] max_diff = {diff:.3e}")
    assert diff < 1e-6, f"apply_rope_shift mismatch: {diff:.3e}"


@pytest.mark.requires_model
@torch.no_grad()
def test_full_reuse_matches_full_recompute_when_only_one_chunk_at_prefix(qwen_setup):
    """Trivial case: a single cached chunk at position 0. Full reuse output must
    equal full recompute output bit-exactly (or within FP32 noise)."""
    tokenizer, lw = qwen_setup

    chunks = chunk_texts(["Hello, world."], tokenizer)
    assert chunks[0].position == 0

    store = KVStore()
    for c in chunks:
        store.put(c.hash, precompute_chunk_kv(lw, c.text, tokenizer))

    fused_logits = fuse_full_reuse(lw, chunks, store)

    full_ids = torch.cat([c.token_ids for c in chunks]).unsqueeze(0)
    recompute_logits = lw.forward_layerwise(full_ids)

    diff = (fused_logits - recompute_logits).abs().max().item()
    print(f"\n[full_reuse 1 chunk] max_diff = {diff:.3e}, S={full_ids.shape[1]}")
    assert diff < 1e-5, f"single-chunk full_reuse should match recompute: {diff:.3e}"


@pytest.mark.requires_model
@torch.no_grad()
def test_full_reuse_diverges_with_multiple_chunks(qwen_setup):
    """Two chunks fused via cached KV must diverge from full recompute, because
    the second chunk's cached K/V was computed without cross-attention to the
    first. This reproduces the paper's documented limitation of full reuse."""
    tokenizer, lw = qwen_setup

    chunks = chunk_texts(
        [
            "The Eiffel Tower is in Paris. ",
            "It was completed in 1889.",
        ],
        tokenizer,
    )
    assert len(chunks) == 2 and chunks[0].position == 0

    store = KVStore()
    for c in chunks:
        store.put(c.hash, precompute_chunk_kv(lw, c.text, tokenizer))

    fused_logits = fuse_full_reuse(lw, chunks, store)

    full_ids = torch.cat([c.token_ids for c in chunks]).unsqueeze(0)
    recompute_logits = lw.forward_layerwise(full_ids)

    diff = fused_logits - recompute_logits
    max_diff = diff.abs().max().item()
    l2 = diff.norm().item()
    boundary = chunks[0].token_ids.shape[0]
    # Slice over the second chunk's positions only — that's where cross-attention
    # was missed.
    second_chunk_l2 = diff[:, boundary:].norm().item()
    print(
        f"\n[full_reuse 2 chunks] max_diff = {max_diff:.3e}, "
        f"L2 (full) = {l2:.3e}, L2 (chunk 2 only) = {second_chunk_l2:.3e}, "
        f"S={full_ids.shape[1]}"
    )

    # Divergence must be measurable. We pick a deliberately loose lower bound
    # (1e-2 in L2 over the whole tensor) since it should be many orders of
    # magnitude above the < 1e-5 noise floor we saw in the single-chunk test.
    assert l2 > 1e-2, f"expected full_reuse to diverge but L2 is only {l2:.3e}"
    # And the divergence should be concentrated in the second chunk's positions.
    assert second_chunk_l2 > 0.5 * l2, (
        "divergence should be concentrated in chunk 2 (it's the one missing "
        f"cross-attention): chunk 2 L2 = {second_chunk_l2:.3e}, total L2 = {l2:.3e}"
    )
