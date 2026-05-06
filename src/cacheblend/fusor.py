"""
Fusor — assembles cached chunk KV into a coherent forward pass.

Phase 2 deliverable: ``fuse_full_reuse``.
Phase 3 deliverable: ``fuse_selective``.

Both functions share the same skeleton:

  for layer in 0 .. L-1:
      register temporary forward hooks on layer.self_attn.{k_proj, v_proj}
      that REPLACE those projections' output with the K, V we want the layer
      to attend over (always **pre-RoPE** — HF then rotates with the fused
      sequence's position_embeddings, so chunk-level RoPE shift is implicit).

The two functions differ only in *what* the hook returns:
  - full reuse: cached K, V at every position.
  - selective: cached K, V at every position EXCEPT the HKVD positions, which
    keep the fresh k_proj / v_proj output (= recomputed under the fused
    sequence's hidden flow).

HKVD selection happens in a small pre-pass at ``check_layer``: we run
input_layernorm + k_proj on the running hidden, apply RoPE manually, and
compare to the cached K (also rotated) to pick the top-r% deviation tokens.
We do **not** re-run the full check_layer — the synth-hook path takes care
of that consistently from layer 0 onwards.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from transformers.cache_utils import DynamicCache

from cacheblend.chunker import Chunk
from cacheblend.hkvd import kv_deviation, select_top_k
from cacheblend.kv_store import KVStore
from cacheblend.model import LayerwiseModel
from cacheblend.rope import _rotate_half


# ---------------------------------------------------------------- helpers

def _kv_to_proj_output(t: Tensor) -> Tensor:
    """(B, num_kv_heads, S, head_dim) → (B, S, num_kv_heads * head_dim).
    Matches the shape that the layer's k_proj / v_proj would have produced.
    """
    B, H, S, D = t.shape
    return t.transpose(1, 2).reshape(B, S, H * D).contiguous()


def _gather_cached_kv_per_layer(
    chunks: Sequence[Chunk],
    kv_store: KVStore,
    num_layers: int,
    device,
    dtype,
) -> Tuple[List[Tensor], List[Tensor]]:
    """Concatenate per-chunk K (pre-RoPE) and V across chunks, per layer.

    Returns:
      (cached_K_per_layer, cached_V_per_layer)
      Each is a list of length num_layers, each entry shape
      (B=1, num_kv_heads, S_total, head_dim).
    """
    ks_per_layer: List[List[Tensor]] = [[] for _ in range(num_layers)]
    vs_per_layer: List[List[Tensor]] = [[] for _ in range(num_layers)]
    for chunk in chunks:
        kv = kv_store.get(chunk.hash)
        if kv is None:
            raise KeyError(f"chunk {chunk.hash[:6]} missing from KVStore")
        for layer_idx, (k_pre, v) in enumerate(kv):
            ks_per_layer[layer_idx].append(k_pre.to(device=device, dtype=dtype))
            vs_per_layer[layer_idx].append(v.to(device=device, dtype=dtype))
    cached_K = [torch.cat(ks, dim=2) for ks in ks_per_layer]
    cached_V = [torch.cat(vs, dim=2) for vs in vs_per_layer]
    return cached_K, cached_V


def _make_synth_hook(
    cached_pre_rope: Tensor,
    keep_fresh_mask: Tensor,
):
    """Return a forward hook that synthesizes a k_proj/v_proj output:

    output[positions where ``keep_fresh_mask`` is True]   = fresh (the actual k_proj/v_proj output)
    output[positions where ``keep_fresh_mask`` is False]  = cached (reshaped to projection shape)

    Args:
      cached_pre_rope:  (B, num_kv_heads, S, head_dim) — what we'd inject if
                        we were doing full reuse.
      keep_fresh_mask:  (S,) bool — True means "let the layer's own
                        projection output through", False means "use cached".
    """
    cached_proj = _kv_to_proj_output(cached_pre_rope)  # (B, S, H*D)
    mask = keep_fresh_mask.to(cached_proj.device).view(1, -1, 1)  # (1, S, 1)

    def hook(_module, _inputs, output: Tensor) -> Tensor:
        return torch.where(mask, output, cached_proj)

    return hook


def _assert_chunks_contiguous(chunks: Sequence[Chunk]) -> None:
    expected = 0
    for c in chunks:
        if c.position != expected:
            raise ValueError(
                f"chunk {c.hash[:6]} has position {c.position}, expected {expected}"
            )
        expected += int(c.token_ids.shape[0])


# --------------------------------------------------------------- full reuse

@torch.no_grad()
def fuse_full_reuse(
    model: LayerwiseModel,
    chunks: Sequence[Chunk],
    kv_store: KVStore,
) -> Tensor:
    """Layerwise forward where K and V at every layer come from the cache.

    Returns logits of shape ``(1, S_total, vocab)``. Used as the Phase 2
    reference and as Phase 3's "ratio = 0" baseline.
    """
    return _fuse(
        model,
        chunks,
        kv_store,
        recompute_ratio=0.0,
        check_layer=0,         # irrelevant when ratio=0
    )


# --------------------------------------------------------------- selective

@torch.no_grad()
def fuse_selective(
    model: LayerwiseModel,
    chunks: Sequence[Chunk],
    kv_store: KVStore,
    recompute_ratio: float = 0.15,
    check_layer: int = 1,
) -> Tensor:
    """CacheBlend selective recompute. Returns logits ``(1, S_total, vocab)``.

    Mechanics:
      - Compute hidden through layers ``0 .. check_layer-1`` using cached
        K, V (full reuse path).
      - At ``check_layer``, run a mini forward (input_layernorm + k_proj +
        RoPE) on the running hidden to materialize fresh K (post-RoPE).
        Compare to cached K (rotated to fused positions). Pick the top
        ``recompute_ratio`` of tokens by squared-L2 deviation → set ``S``.
      - Run all layers ``0 .. L-1`` with a synth hook that returns:
        * the layer's own fresh k_proj / v_proj output at positions in ``S``
        * cached pre-RoPE K, V at positions outside ``S``
        For ``ratio == 0`` this collapses to full reuse; for ``ratio == 1``
        it collapses to full recompute (because layer-0 cached pre-RoPE K
        already equals fresh K at layer 0 — embeddings are position-agnostic).

    Note: ``hkvd_indices`` from ``check_layer`` is reused at every later
    layer (no gradual narrowing in v1, matching LMCache's actual behavior).
    """
    return _fuse(
        model,
        chunks,
        kv_store,
        recompute_ratio=float(recompute_ratio),
        check_layer=int(check_layer),
    )


# ----------------------------------------------------------------- internal

@torch.no_grad()
def _fuse(
    model: LayerwiseModel,
    chunks: Sequence[Chunk],
    kv_store: KVStore,
    recompute_ratio: float,
    check_layer: int,
) -> Tensor:
    if not chunks:
        raise ValueError("chunks is empty")
    _assert_chunks_contiguous(chunks)

    device = model.device
    dtype = model.dtype

    full_ids = torch.cat([c.token_ids for c in chunks]).unsqueeze(0).to(device)
    B, S_total = full_ids.shape

    hidden = model.embed_tokens(full_ids)
    pos_ids = torch.arange(S_total, device=device).unsqueeze(0).expand(B, -1)
    cache_pos = torch.arange(S_total, device=device)
    pe = model.compute_position_embeddings(hidden, pos_ids)
    layer_cache = DynamicCache(config=model.config)
    mask = model.build_causal_mask(hidden, pos_ids, cache_pos, layer_cache)

    cached_K, cached_V = _gather_cached_kv_per_layer(
        chunks, kv_store, model.num_layers, device, dtype
    )

    hkvd_indices = _select_hkvd_at_check_layer(
        model=model,
        hidden_at_layer_0=hidden,
        cached_K=cached_K,
        cached_V=cached_V,
        pos_ids=pos_ids,
        cache_pos=cache_pos,
        pe=pe,
        attn_mask=mask,
        check_layer=check_layer,
        recompute_ratio=recompute_ratio,
        S_total=S_total,
    )

    # _select_hkvd_at_check_layer used its own throwaway cache; the main loop
    # below needs a fresh DynamicCache so cache.layers[0..check_layer-1] start
    # empty (avoids attention seeing 2× tokens via past_key_values.update concat).
    layer_cache = DynamicCache(config=model.config)
    mask = model.build_causal_mask(hidden, pos_ids, cache_pos, layer_cache)

    # Build the per-position "let fresh through" mask used by every layer's
    # synth hook. For Phase 3 this is the same set everywhere; Phase 5 may
    # narrow per-layer.
    keep_fresh = torch.zeros(S_total, dtype=torch.bool, device=device)
    if hkvd_indices.numel() > 0:
        keep_fresh[hkvd_indices.to(device)] = True

    for layer_idx in range(model.num_layers):
        layer = model._inner.layers[layer_idx]
        kproj = layer.self_attn.k_proj
        vproj = layer.self_attn.v_proj

        kh = kproj.register_forward_hook(_make_synth_hook(cached_K[layer_idx], keep_fresh))
        vh = vproj.register_forward_hook(_make_synth_hook(cached_V[layer_idx], keep_fresh))
        try:
            out = model.prefill_layer(
                layer_idx=layer_idx,
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

    return model.final_norm_and_lm_head(hidden)


@torch.no_grad()
def _select_hkvd_at_check_layer(
    *,
    model: LayerwiseModel,
    hidden_at_layer_0: Tensor,
    cached_K,
    cached_V,
    pos_ids: Tensor,
    cache_pos: Tensor,
    pe: Tuple[Tensor, Tensor],
    attn_mask,
    check_layer: int,
    recompute_ratio: float,
    S_total: int,
) -> Tensor:
    """Run layers 0..check_layer-1 in full reuse mode, then a mini forward at
    check_layer (input_layernorm + k_proj + RoPE) to compute fresh K. Return
    indices of the top-``recompute_ratio`` tokens by squared-L2 deviation.

    Uses its own throwaway ``DynamicCache`` so the caller's cache stays empty.
    """
    if recompute_ratio <= 0.0:
        return torch.empty(0, dtype=torch.long)
    if recompute_ratio >= 1.0:
        return torch.arange(S_total, dtype=torch.long)

    layer_cache = DynamicCache(config=model.config)

    # Walk hidden through layers [0, check_layer) using full reuse.
    hidden = hidden_at_layer_0
    keep_fresh_empty = torch.zeros(S_total, dtype=torch.bool, device=hidden.device)
    for layer_idx in range(check_layer):
        layer = model._inner.layers[layer_idx]
        kproj = layer.self_attn.k_proj
        vproj = layer.self_attn.v_proj
        kh = kproj.register_forward_hook(_make_synth_hook(cached_K[layer_idx], keep_fresh_empty))
        vh = vproj.register_forward_hook(_make_synth_hook(cached_V[layer_idx], keep_fresh_empty))
        try:
            out = model.prefill_layer(
                layer_idx=layer_idx,
                hidden_states=hidden,
                position_ids=pos_ids,
                position_embeddings=pe,
                attention_mask=attn_mask,
                past_key_values=layer_cache,
                cache_position=cache_pos,
            )
            hidden = out.hidden
        finally:
            kh.remove()
            vh.remove()

    # Mini forward at check_layer: input_layernorm + k_proj, then RoPE.
    layer = model._inner.layers[check_layer]
    normed = layer.input_layernorm(hidden)
    fresh_k_proj = layer.self_attn.k_proj(normed)
    B, S_, _ = fresh_k_proj.shape
    fresh_k_pre = (
        fresh_k_proj.view(B, S_, model.num_kv_heads, model.head_dim)
        .transpose(1, 2)
        .contiguous()
    )

    cos, sin = pe
    cos_b = cos.unsqueeze(1)
    sin_b = sin.unsqueeze(1)
    fresh_k_post = (fresh_k_pre * cos_b) + (_rotate_half(fresh_k_pre) * sin_b)
    cached_k_pre = cached_K[check_layer]
    cached_k_post = (cached_k_pre * cos_b) + (_rotate_half(cached_k_pre) * sin_b)

    deviation = kv_deviation(fresh_k_post, cached_k_post)
    k = max(1, int(round(S_total * recompute_ratio)))
    return select_top_k(deviation, k)
