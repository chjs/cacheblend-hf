"""
Fusor — assembles cached chunk KV into a coherent forward pass.

Phase 2 deliverable: ``fuse_full_reuse``. Phase 3 will add ``fuse_selective``
on top of the same skeleton.

Strategy (full reuse):

For each layer ``i`` of the fused forward pass:
1. Concatenate the cached **pre-RoPE** K and V of every chunk along the
   sequence dimension. Shape: ``(B=1, num_kv_heads, S_total, head_dim)``.
2. Register temporary forward hooks on the layer's ``k_proj`` and ``v_proj``
   that **return** the cached tensors, replacing the projections' output.
   The decoder layer then does its normal RoPE / attention / o_proj / MLP
   flow — but K and V come from the cache instead of the current hidden.
3. The fresh hidden state still flows layer-to-layer normally, so Q is
   computed from the actual evolving hidden. RoPE is applied by the layer
   on our injected pre-RoPE K using the **fused** sequence's
   position_embeddings, which is exactly the per-chunk position shift we
   want.

Why this gives full reuse:
- Single chunk at prefix → cached K/V at this layer were computed with the
  same hidden flow as the fused forward (no cross-chunk attention possible)
  → injected K/V == fresh K/V → result bit-identical to full recompute.
- Multi chunks → later chunks' cached K/V were computed with their own solo
  hidden flow, missing cross-attention to earlier chunks → injected K/V
  differs from fresh K/V → measurable logit divergence (the paper's expected
  behavior; Phase 3 will recover most of this via selective recompute).
"""
from __future__ import annotations

from typing import List, Sequence

import torch
from torch import Tensor
from transformers.cache_utils import DynamicCache

from cacheblend.chunker import Chunk
from cacheblend.kv_store import KVStore
from cacheblend.model import LayerwiseModel


def _kv_to_proj_output(t: Tensor) -> Tensor:
    """Reshape (B, num_kv_heads, S, head_dim) to k_proj/v_proj output shape (B, S, kv_heads * head_dim)."""
    B, H, S, D = t.shape
    return t.transpose(1, 2).reshape(B, S, H * D).contiguous()


@torch.no_grad()
def fuse_full_reuse(
    model: LayerwiseModel,
    chunks: Sequence[Chunk],
    kv_store: KVStore,
) -> Tensor:
    """Run a layerwise forward where K and V at every layer come from the cache.

    Returns logits of shape ``(1, S_total, vocab)``.

    All chunks must already be cached in ``kv_store``. Behavior on a miss is to
    raise — Phase 4 will add prefetch / on-demand precompute.
    """
    device = model.device
    dtype = model.dtype

    # Concatenate token ids in chunk order.
    full_ids = torch.cat([c.token_ids for c in chunks]).unsqueeze(0).to(device)
    B, S_total = full_ids.shape

    # Sanity: chunks must report contiguous, monotonic positions starting at 0.
    expected_pos = 0
    for c in chunks:
        if c.position != expected_pos:
            raise ValueError(
                f"chunk {c.hash[:6]} has position {c.position}, expected {expected_pos}"
            )
        expected_pos += int(c.token_ids.shape[0])

    hidden = model.embed_tokens(full_ids)
    pos_ids = torch.arange(S_total, device=device).unsqueeze(0).expand(B, -1)
    cache_pos = torch.arange(S_total, device=device)
    pe = model.compute_position_embeddings(hidden, pos_ids)
    layer_cache = DynamicCache(config=model.config)
    mask = model.build_causal_mask(hidden, pos_ids, cache_pos, layer_cache)

    for layer_idx in range(model.num_layers):
        # Gather cached pre-RoPE K and V for this layer across all chunks.
        ks: List[Tensor] = []
        vs: List[Tensor] = []
        for chunk in chunks:
            kv = kv_store.get(chunk.hash)
            if kv is None:
                raise KeyError(f"chunk {chunk.hash[:6]!s} missing from KVStore")
            k_pre, v = kv[layer_idx]
            ks.append(k_pre.to(device=device, dtype=dtype))
            vs.append(v.to(device=device, dtype=dtype))
        k_cat = torch.cat(ks, dim=2)   # (B, kv_heads, S_total, head_dim)
        v_cat = torch.cat(vs, dim=2)

        layer = model._inner.layers[layer_idx]
        kproj = layer.self_attn.k_proj
        vproj = layer.self_attn.v_proj

        k_override = _kv_to_proj_output(k_cat)
        v_override = _kv_to_proj_output(v_cat)

        def _make_override(repl: Tensor):
            def hook(_module, _inputs, _output):
                return repl
            return hook

        kh = kproj.register_forward_hook(_make_override(k_override))
        vh = vproj.register_forward_hook(_make_override(v_override))
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
