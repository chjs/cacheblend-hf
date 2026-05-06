"""
Per-chunk KV pre-computation.

Phase 2 deliverable. Runs a chunk through the model alone (positions
``0 .. S_chunk-1``) and returns per-layer **pre-RoPE** K and V. The Fusor will
RoPE-shift K to the chunk's target position at fuse time.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor
from transformers.cache_utils import DynamicCache

from cacheblend.model import LayerwiseModel


@torch.no_grad()
def precompute_chunk_kv(
    model: LayerwiseModel,
    chunk_text: str,
    tokenizer,
) -> List[Tuple[Tensor, Tensor]]:
    """Run a full layerwise prefill on the chunk text alone, return per-layer (K, V).

    K is returned in the form configured on the model (default: ``pre_rope``).
    """
    ids = tokenizer(
        chunk_text, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(model.device)

    return precompute_chunk_kv_from_ids(model, ids)


@torch.no_grad()
def precompute_chunk_kv_from_ids(
    model: LayerwiseModel,
    input_ids: Tensor,
) -> List[Tuple[Tensor, Tensor]]:
    """Same as ``precompute_chunk_kv`` but operating on already-tokenized ids."""
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    B, S = input_ids.shape

    hidden = model.embed_tokens(input_ids)
    pos_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
    cache_pos = torch.arange(S, device=input_ids.device)
    pe = model.compute_position_embeddings(hidden, pos_ids)
    cache = DynamicCache(config=model.config)
    mask = model.build_causal_mask(hidden, pos_ids, cache_pos, cache)

    kv_per_layer: List[Tuple[Tensor, Tensor]] = []
    for i in range(model.num_layers):
        out = model.prefill_layer(
            layer_idx=i,
            hidden_states=hidden,
            position_ids=pos_ids,
            position_embeddings=pe,
            attention_mask=mask,
            past_key_values=cache,
            cache_position=cache_pos,
        )
        hidden = out.hidden
        kv_per_layer.append((out.k.detach(), out.v.detach()))
    return kv_per_layer
