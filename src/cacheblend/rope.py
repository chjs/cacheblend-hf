"""
RoPE position shift — apply rotation to a pre-RoPE K so it lives at new positions.

Phase 2 deliverable.

Convention: HF Llama / Qwen2 / Mistral all use the **rotate_half** style
(K's last dim split into two halves, rotated as a 2D vector). We replicate
that exact computation here, sourcing (cos, sin) from the model's own
``rotary_emb`` module via ``LayerwiseModel.compute_position_embeddings``. This
guarantees we never disagree with the model on rotation conventions, and it
also picks up RoPE scaling / theta from the model's config.
"""
from __future__ import annotations

import torch
from torch import Tensor


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_shift(
    k_pre_rope: Tensor,
    target_positions: Tensor,
    model,
) -> Tensor:
    """Rotate ``k_pre_rope`` so it represents K at ``target_positions``.

    Args:
      k_pre_rope: (B, num_kv_heads, S, head_dim) — K *before* any rotation.
      target_positions: (S,) int — absolute positions to rotate to.
      model: a ``LayerwiseModel`` whose ``compute_position_embeddings`` returns
             (cos, sin) compatible with ``apply_rotary_pos_emb``.

    Returns:
      (B, num_kv_heads, S, head_dim) — K *after* rotation to the target positions.
    """
    if target_positions.dim() != 1:
        raise ValueError(
            f"target_positions must be 1D (got shape {tuple(target_positions.shape)})"
        )
    B, _, S, _ = k_pre_rope.shape
    if int(target_positions.shape[0]) != S:
        raise ValueError(
            f"target_positions length {int(target_positions.shape[0])} != K seq len {S}"
        )

    # Build (cos, sin) of shape (B, S, head_dim). The model's rotary_emb only
    # uses hidden_states for shape/dtype/device; values don't matter.
    dummy_hidden = torch.zeros(
        B, S, model.hidden_size, dtype=k_pre_rope.dtype, device=k_pre_rope.device
    )
    pos_ids = target_positions.to(device=k_pre_rope.device).unsqueeze(0).expand(B, -1)
    cos, sin = model.compute_position_embeddings(dummy_hidden, pos_ids)

    # cos/sin: (B, S, head_dim). Insert a head dim for broadcast.
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    return (k_pre_rope * cos) + (_rotate_half(k_pre_rope) * sin)
