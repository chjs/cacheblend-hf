"""
LayerwiseModel: HF Transformers wrapper exposing per-layer prefill.

Phase 1 deliverable. Verified to be **bit-exact** (FP32, max_diff < 1e-5) against
the standard ``model(...).logits`` for Qwen2-family architectures.

Design notes (read once):

- Works on any HF causal LM whose top-level module exposes
  ``model.model.embed_tokens``, ``model.model.layers`` (a list of decoder layers
  whose ``forward`` matches the Llama/Qwen2/Mistral signature with
  ``position_embeddings=(cos, sin)``), ``model.model.rotary_emb``,
  ``model.model.norm``, and ``model.lm_head``. This covers Llama, Qwen2, Qwen2.5,
  and Mistral.
- This wrapper does NOT contain CacheBlend logic. It only exposes a per-layer
  prefill interface so later phases can intercept K/V between layers.
- KV is captured in **two parallel forms** per layer:
    * pre-RoPE K (via a forward hook on ``k_proj``)
    * post-RoPE K + V (read from ``DynamicCache`` after the layer call)
  ``prefill_layer`` returns the form selected by ``self.kv_form``. The default
  ``"pre_rope"`` is the convention chosen in Phase 0; if a future arch breaks
  the hook contract, switch to ``"post_rope"`` and adjust Phase 2's RoPE shift
  to apply the inverse rotation first.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask


KVForm = Literal["pre_rope", "post_rope"]


@dataclass
class LayerOutput:
    """Per-layer prefill output. Shapes assume batch_first."""
    hidden: Tensor              # (B, S, hidden)
    k: Tensor                   # (B, num_kv_heads, S, head_dim)
    v: Tensor                   # (B, num_kv_heads, S, head_dim)


class LayerwiseModel:
    """Wraps a HuggingFace causal LM to expose per-layer prefill.

    Usage:
        lw = LayerwiseModel("Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.float32, device="cpu")
        hidden = lw.embed_tokens(input_ids)
        position_ids = torch.arange(input_ids.shape[1])[None, :]
        position_embeddings = lw.compute_position_embeddings(hidden, position_ids)
        for i in range(lw.num_layers):
            hidden, k, v = lw.prefill_layer(i, hidden, position_ids, position_embeddings)
        logits = lw.final_norm_and_lm_head(hidden)
    """

    def __init__(
        self,
        hf_model_or_id: Union[str, PreTrainedModel],
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        kv_form: KVForm = "pre_rope",
    ):
        if isinstance(hf_model_or_id, str):
            model = AutoModelForCausalLM.from_pretrained(
                hf_model_or_id, torch_dtype=dtype
            )
        else:
            model = hf_model_or_id
            if dtype is not None:
                model = model.to(dtype=dtype)

        model = model.to(device)
        model.eval()

        self.hf_model: PreTrainedModel = model
        self.config = model.config
        self.device = torch.device(device)
        self.dtype = dtype
        self.kv_form: KVForm = kv_form

        # Resolve the inner causal model. HF wraps it under ``.model`` for
        # CausalLM heads; some archs may differ.
        inner = getattr(model, "model", None)
        if inner is None or not hasattr(inner, "layers"):
            raise ValueError(
                f"Unsupported architecture: {type(model).__name__} has no "
                f".model.layers. Add an arch-specific resolver in LayerwiseModel."
            )
        self._inner = inner

        # Cache pre-RoPE K captured by k_proj hooks during the most recent
        # decoder_layer call. Keyed by layer_idx -> (B, num_kv_heads, S, head_dim).
        self._pre_rope_k: dict[int, Tensor] = {}
        self._install_k_proj_hooks()

    # ------------------------------------------------------------------ Properties

    @property
    def num_layers(self) -> int:
        return self.config.num_hidden_layers

    @property
    def num_kv_heads(self) -> int:
        # Llama/Qwen2/Mistral all expose this attribute.
        return self.config.num_key_value_heads

    @property
    def num_attention_heads(self) -> int:
        return self.config.num_attention_heads

    @property
    def head_dim(self) -> int:
        return getattr(
            self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads
        )

    @property
    def rope_theta(self) -> float:
        return float(getattr(self.config, "rope_theta", 10000.0))

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    # ------------------------------------------------------------------ Public API

    def embed_tokens(self, input_ids: Tensor) -> Tensor:
        """input_ids: (B, S) -> hidden: (B, S, hidden_size)."""
        return self._inner.embed_tokens(input_ids)

    def compute_position_embeddings(
        self, hidden_states: Tensor, position_ids: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Run the model's RoPE block once to get (cos, sin) shared across layers."""
        return self._inner.rotary_emb(hidden_states, position_ids)

    def build_causal_mask(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
        cache_position: Tensor,
        past_key_values: Optional[DynamicCache] = None,
    ):
        """Build the same causal mask the standard forward would build."""
        return create_causal_mask(
            config=self.config,
            input_embeds=hidden_states,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

    def prefill_layer(
        self,
        layer_idx: int,
        hidden_states: Tensor,
        position_ids: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        attention_mask=None,
        past_key_values: Optional[DynamicCache] = None,
        cache_position: Optional[Tensor] = None,
    ) -> LayerOutput:
        """Run one decoder layer; return new hidden + (K, V).

        K is returned in the form chosen by ``self.kv_form`` (default pre-RoPE).
        V is unchanged by RoPE.
        """
        if cache_position is None:
            S = hidden_states.shape[1]
            cache_position = torch.arange(S, device=hidden_states.device)

        if past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if attention_mask is None:
            attention_mask = self.build_causal_mask(
                hidden_states, position_ids, cache_position, past_key_values
            )

        # Clear pre-RoPE capture slot for this layer; the hook will repopulate.
        self._pre_rope_k.pop(layer_idx, None)

        layer = self._inner.layers[layer_idx]
        new_hidden = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        post_rope_k, v = self._read_cache_layer(past_key_values, layer_idx)
        if self.kv_form == "pre_rope":
            k = self._pre_rope_k.pop(layer_idx, None)
            if k is None:
                raise RuntimeError(
                    f"Pre-RoPE K not captured for layer {layer_idx}. "
                    f"Forward hook on k_proj did not fire — set kv_form='post_rope' "
                    f"or implement an arch-specific capture path."
                )
        else:
            k = post_rope_k

        return LayerOutput(hidden=new_hidden, k=k, v=v)

    def final_norm_and_lm_head(self, hidden: Tensor) -> Tensor:
        """hidden: (B, S, H) -> logits: (B, S, vocab)."""
        hidden = self._inner.norm(hidden)
        logits = self.hf_model.lm_head(hidden)
        return logits

    @torch.no_grad()
    def forward_layerwise(self, input_ids: Tensor) -> Tensor:
        """Convenience: run the full layerwise prefill and return logits.

        Used by tests to compare against ``hf_model(input_ids).logits``.
        """
        B, S = input_ids.shape
        device = input_ids.device

        hidden = self.embed_tokens(input_ids)
        position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        cache_position = torch.arange(S, device=device)

        position_embeddings = self.compute_position_embeddings(hidden, position_ids)
        cache = DynamicCache(config=self.config)
        attn_mask = self.build_causal_mask(hidden, position_ids, cache_position, cache)

        for i in range(self.num_layers):
            out = self.prefill_layer(
                layer_idx=i,
                hidden_states=hidden,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                attention_mask=attn_mask,
                past_key_values=cache,
                cache_position=cache_position,
            )
            hidden = out.hidden

        return self.final_norm_and_lm_head(hidden)

    # ------------------------------------------------------------------ Internals

    def _install_k_proj_hooks(self) -> None:
        """Capture pre-RoPE K via a forward hook on each layer's k_proj."""
        for i, layer in enumerate(self._inner.layers):
            kproj = layer.self_attn.k_proj
            kproj.register_forward_hook(self._make_k_hook(i))

    def _make_k_hook(self, layer_idx: int):
        kv_heads = self.num_kv_heads
        head_dim = self.head_dim

        def hook(module, inputs, output: Tensor):
            # k_proj output: (B, S, num_kv_heads * head_dim).
            B, S, _ = output.shape
            k = output.view(B, S, kv_heads, head_dim).transpose(1, 2).contiguous()
            self._pre_rope_k[layer_idx] = k

        return hook

    def _read_cache_layer(
        self, cache: DynamicCache, layer_idx: int
    ) -> Tuple[Tensor, Tensor]:
        """Pull the K, V tensors stored by ``cache`` at the given layer."""
        layers = getattr(cache, "layers", None)
        if layers is not None and layer_idx < len(layers):
            entry = layers[layer_idx]
            return entry.keys, entry.values
        # Fallback for older API (key_cache/value_cache lists).
        if hasattr(cache, "key_cache"):
            return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
        raise RuntimeError(
            f"Cannot read KV from cache type {type(cache).__name__} at layer {layer_idx}"
        )
