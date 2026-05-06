"""
LayerwiseModel: HF Transformers wrapper exposing per-layer prefill.

PHASE 1: implement this module.

The goal is to make the model callable layer-by-layer, with the same numerical
output as a single `model(...)` call.

NOT a CacheBlend module — just an enabling abstraction. Keep CacheBlend logic
out of this file.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


class LayerwiseModel:
    """Wraps a HuggingFace causal LM to expose per-layer prefill.

    Properties:
        num_layers: number of decoder layers
        num_kv_heads: K/V heads (≠ num_attention_heads when GQA)
        head_dim: per-head dimension
        rope_theta: RoPE base frequency
        device, dtype
    """

    def __init__(
        self,
        hf_model_or_id,           # Union[str, PreTrainedModel]
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        # TODO Phase 1:
        # 1. Load model + tokenizer if string given
        # 2. Move to device, set dtype
        # 3. Extract: embed_tokens, decoder layers, final norm, lm_head, rotary_emb
        # 4. Store config: num_layers, num_kv_heads, head_dim, rope_theta
        raise NotImplementedError("Phase 1: implement LayerwiseModel.__init__")

    @property
    def num_layers(self) -> int:
        raise NotImplementedError

    @property
    def num_kv_heads(self) -> int:
        raise NotImplementedError

    @property
    def head_dim(self) -> int:
        raise NotImplementedError

    @property
    def rope_theta(self) -> float:
        raise NotImplementedError

    def embed_tokens(self, input_ids: Tensor) -> Tensor:
        """input_ids: (B, S) -> hidden: (B, S, hidden)."""
        raise NotImplementedError("Phase 1")

    def prefill_layer(
        self,
        layer_idx: int,
        hidden_states: Tensor,             # (B, S, H)
        position_ids: Tensor,              # (B, S)
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Run one decoder layer.

        Returns: (new_hidden, (k, v))
            new_hidden: (B, S, H)
            k, v: (B, num_kv_heads, S, head_dim)
        """
        raise NotImplementedError("Phase 1")

    def final_norm_and_lm_head(self, hidden: Tensor) -> Tensor:
        """hidden: (B, S, H) -> logits: (B, S, vocab)."""
        raise NotImplementedError("Phase 1")

    # ------- Phase 3 extension (partial prefill) -------
    def prefill_layer_partial(
        self,
        layer_idx: int,
        hidden_states: Tensor,
        position_ids: Tensor,
        cached_kv: Tuple[Tensor, Tensor],
        recompute_indices: Tensor,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """See ARCHITECTURE.md and tasks/phase-3-selective-recompute.md."""
        raise NotImplementedError("Phase 3")
