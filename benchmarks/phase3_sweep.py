"""
Phase 3 sweep — used to fill the report's ratio/L2 table and Insight-2 overlap.

Run once locally; this script is not part of pytest.

  python benchmarks/phase3_sweep.py

Output: prints a Markdown-ready table to stdout.
"""
from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from cacheblend.chunker import chunk_texts
from cacheblend.fusor import (
    _gather_cached_kv_per_layer,
    _make_synth_hook,
    fuse_full_reuse,
    fuse_selective,
)
from cacheblend.hkvd import select_top_k
from cacheblend.kv_store import KVStore
from cacheblend.model import LayerwiseModel
from cacheblend.precompute import precompute_chunk_kv
from cacheblend.rope import _rotate_half


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
TEXTS = [
    "The Eiffel Tower is in Paris. ",
    "It was completed in 1889.",
]
RATIOS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.50, 1.0]


def _deviation_at_layer(
    lw: LayerwiseModel,
    chunks,
    kv_store: KVStore,
    target_layer: int,
) -> torch.Tensor:
    """Return per-token squared-L2 deviation between fresh and cached K at
    ``target_layer``, after running full reuse through ``target_layer - 1``."""
    device = lw.device
    dtype = lw.dtype
    full_ids = torch.cat([c.token_ids for c in chunks]).unsqueeze(0).to(device)
    B, S = full_ids.shape

    hidden = lw.embed_tokens(full_ids)
    pos_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    cache_pos = torch.arange(S, device=device)
    pe = lw.compute_position_embeddings(hidden, pos_ids)
    layer_cache = DynamicCache(config=lw.config)
    mask = lw.build_causal_mask(hidden, pos_ids, cache_pos, layer_cache)

    cached_K, cached_V = _gather_cached_kv_per_layer(
        chunks, kv_store, lw.num_layers, device, dtype
    )

    keep_empty = torch.zeros(S, dtype=torch.bool, device=device)
    for li in range(target_layer):
        layer = lw._inner.layers[li]
        kh = layer.self_attn.k_proj.register_forward_hook(
            _make_synth_hook(cached_K[li], keep_empty)
        )
        vh = layer.self_attn.v_proj.register_forward_hook(
            _make_synth_hook(cached_V[li], keep_empty)
        )
        try:
            out = lw.prefill_layer(
                layer_idx=li,
                hidden_states=hidden,
                position_ids=pos_ids,
                position_embeddings=pe,
                attention_mask=mask,
                past_key_values=layer_cache,
                cache_position=cache_pos,
            )
            hidden = out.hidden
        finally:
            kh.remove()
            vh.remove()

    layer = lw._inner.layers[target_layer]
    normed = layer.input_layernorm(hidden)
    fresh_proj = layer.self_attn.k_proj(normed)
    B, S_, _ = fresh_proj.shape
    fresh_pre = (
        fresh_proj.view(B, S_, lw.num_kv_heads, lw.head_dim).transpose(1, 2).contiguous()
    )
    cos, sin = pe
    cos_b, sin_b = cos.unsqueeze(1), sin.unsqueeze(1)
    fresh_post = (fresh_pre * cos_b) + (_rotate_half(fresh_pre) * sin_b)
    cached_pre = cached_K[target_layer]
    cached_post = (cached_pre * cos_b) + (_rotate_half(cached_pre) * sin_b)

    diff = fresh_post.float() - cached_post.float()
    return (diff * diff).sum(dim=(1, 3))[0]


@torch.no_grad()
def main():
    print(f"# Phase 3 Sweep — {MODEL_ID}, FP32 CPU")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, attn_implementation="eager"
    ).to("cpu").eval()
    lw = LayerwiseModel(hf, dtype=torch.float32, device="cpu", kv_form="pre_rope")
    print(f"loaded in {time.time() - t0:.1f}s")

    chunks = chunk_texts(TEXTS, tokenizer)
    store = KVStore()
    for c in chunks:
        store.put(c.hash, precompute_chunk_kv(lw, c.text, tokenizer))
    full_ids = torch.cat([c.token_ids for c in chunks]).unsqueeze(0)
    S_total = int(full_ids.shape[1])
    print(f"S_total = {S_total} (chunks: {[int(c.token_ids.shape[0]) for c in chunks]})")

    t = time.time()
    recompute = lw.forward_layerwise(full_ids)
    print(f"recompute baseline computed in {time.time() - t:.1f}s")

    t = time.time()
    full_reuse = fuse_full_reuse(lw, chunks, store)
    l2_reuse = (full_reuse - recompute).norm().item()
    max_reuse = (full_reuse - recompute).abs().max().item()
    print(f"full_reuse computed in {time.time() - t:.1f}s; L2 = {l2_reuse:.3e}, max = {max_reuse:.3e}")

    print()
    print("| ratio | L2 vs recompute | max_diff | L2 / L2(reuse) |")
    print("|---|---|---|---|")
    for r in RATIOS:
        t = time.time()
        sel = fuse_selective(lw, chunks, store, recompute_ratio=r, check_layer=1)
        l2 = (sel - recompute).norm().item()
        md = (sel - recompute).abs().max().item()
        ratio_to_reuse = l2 / l2_reuse if l2_reuse > 0 else float("nan")
        print(
            f"| {r:.2f} | {l2:.3e} | {md:.3e} | {ratio_to_reuse:.3f} | "
            f"({time.time() - t:.1f}s)"
        )

    # Insight 2: HKVD overlap between layer 1 and a late layer (here L-1).
    print("\n## Insight 2 — HKVD overlap across layers")
    k = max(1, int(round(S_total * 0.15)))
    print(f"target ratio = 0.15 → k = {k}")

    dev_layer1 = _deviation_at_layer(lw, chunks, store, target_layer=1)
    s1 = set(select_top_k(dev_layer1, k).tolist())
    last = lw.num_layers - 1
    dev_layer_last = _deviation_at_layer(lw, chunks, store, target_layer=last)
    s_last = set(select_top_k(dev_layer_last, k).tolist())
    overlap = len(s1 & s_last) / k
    print(f"layer 1 top-{k}: {sorted(s1)}")
    print(f"layer {last} top-{k}: {sorted(s_last)}")
    print(f"overlap = {len(s1 & s_last)}/{k} = {overlap:.3f}")

    print(f"\ntotal: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
