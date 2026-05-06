"""Phase 1 tests: layerwise forward must be bit-exact vs standard forward."""
from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cacheblend.model import LayerwiseModel


# Loading a 1.5B model for every test is slow; cache via session scope.
@pytest.fixture(scope="module")
def qwen_setup():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, attn_implementation="eager"
    ).to("cpu").eval()
    return model_id, tokenizer, hf_model


def _ids(tokenizer, text: str) -> torch.Tensor:
    return tokenizer(text, return_tensors="pt").input_ids


@pytest.mark.requires_model
@torch.no_grad()
def test_layerwise_matches_standard(qwen_setup):
    """Per-layer prefill must reproduce standard ``model(...).logits`` bit-exactly."""
    model_id, tokenizer, hf_model = qwen_setup

    # Reuse the loaded model so the wrapper's hooks attach to the same instance.
    lw = LayerwiseModel(hf_model, dtype=torch.float32, device="cpu", kv_form="pre_rope")

    input_ids = _ids(tokenizer, "Phase one verification.")

    std_logits = hf_model(input_ids=input_ids, use_cache=False).logits
    lw_logits = lw.forward_layerwise(input_ids)

    max_diff = (std_logits - lw_logits).abs().max().item()
    print(f"\n[bit-exact] max_diff = {max_diff:.3e}")
    assert max_diff < 1e-5, f"layerwise vs standard logit mismatch: {max_diff:.3e}"


@pytest.mark.slow
@pytest.mark.requires_model
@torch.no_grad()
def test_layerwise_matches_standard_longer(qwen_setup):
    """Stress with a longer context to amplify any per-layer drift.

    Marked ``slow`` because CPU FP32 attention is O(S²) and a 150-token forward
    on Qwen2.5-1.5B takes minutes per pass. Skipped by default; run with
    ``pytest -m slow``.
    """
    model_id, tokenizer, hf_model = qwen_setup
    lw = LayerwiseModel(hf_model, dtype=torch.float32, device="cpu", kv_form="pre_rope")

    text = (
        "CacheBlend partitions the input into chunks, precomputes per-chunk KV, "
        "and recomputes a small fraction of high-deviation tokens at inference. "
    ) * 2  # ~60-80 tokens
    input_ids = _ids(tokenizer, text)

    std_logits = hf_model(input_ids=input_ids, use_cache=False).logits
    lw_logits = lw.forward_layerwise(input_ids)

    max_diff = (std_logits - lw_logits).abs().max().item()
    print(f"\n[bit-exact long] max_diff = {max_diff:.3e}, S={input_ids.shape[1]}")
    assert max_diff < 1e-5, f"layerwise vs standard logit mismatch: {max_diff:.3e}"


@pytest.mark.requires_model
@torch.no_grad()
def test_kv_extraction(qwen_setup):
    """The KV captured by LayerwiseModel must match what HF stores in DynamicCache.

    With ``kv_form='post_rope'``, the captured K must equal HF's cached K exactly.
    With ``kv_form='pre_rope'``, applying RoPE to the captured K must reproduce
    HF's cached K (proves the hook captured the value just before rotation).
    """
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    model_id, tokenizer, hf_model = qwen_setup
    input_ids = _ids(tokenizer, "Phase 1 verification: KV extraction.")

    # Reference: capture HF's cache via standard forward with use_cache=True.
    ref_cache = DynamicCache(config=hf_model.config)
    hf_model(input_ids=input_ids, use_cache=True, past_key_values=ref_cache)

    # ---- post-RoPE form ----
    lw_post = LayerwiseModel(hf_model, dtype=torch.float32, device="cpu", kv_form="post_rope")
    B, S = input_ids.shape
    hidden = lw_post.embed_tokens(input_ids)
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    cache_position = torch.arange(S)
    pe = lw_post.compute_position_embeddings(hidden, position_ids)
    cache = DynamicCache(config=hf_model.config)
    attn_mask = lw_post.build_causal_mask(hidden, position_ids, cache_position, cache)

    for i in range(lw_post.num_layers):
        out = lw_post.prefill_layer(
            layer_idx=i,
            hidden_states=hidden,
            position_ids=position_ids,
            position_embeddings=pe,
            attention_mask=attn_mask,
            past_key_values=cache,
            cache_position=cache_position,
        )
        hidden = out.hidden
        ref_k = ref_cache.layers[i].keys
        ref_v = ref_cache.layers[i].values
        assert torch.equal(out.k, ref_k), f"post-RoPE K mismatch at layer {i}"
        assert torch.equal(out.v, ref_v), f"V mismatch at layer {i}"

    # ---- pre-RoPE form ----
    lw_pre = LayerwiseModel(hf_model, dtype=torch.float32, device="cpu", kv_form="pre_rope")
    hidden = lw_pre.embed_tokens(input_ids)
    pe = lw_pre.compute_position_embeddings(hidden, position_ids)
    cache = DynamicCache(config=hf_model.config)
    attn_mask = lw_pre.build_causal_mask(hidden, position_ids, cache_position, cache)
    cos, sin = pe

    for i in range(lw_pre.num_layers):
        out = lw_pre.prefill_layer(
            layer_idx=i,
            hidden_states=hidden,
            position_ids=position_ids,
            position_embeddings=pe,
            attention_mask=attn_mask,
            past_key_values=cache,
            cache_position=cache_position,
        )
        hidden = out.hidden
        # Apply RoPE to the captured pre-RoPE K and compare against HF's cached K.
        # apply_rotary_pos_emb returns (q_rot, k_rot); we only care about k.
        dummy_q = torch.zeros_like(out.k)
        _, k_rot = apply_rotary_pos_emb(dummy_q, out.k, cos, sin)
        ref_k = ref_cache.layers[i].keys
        max_diff = (k_rot - ref_k).abs().max().item()
        assert max_diff < 1e-6, f"pre-RoPE K does not rotate to HF cache at layer {i}: {max_diff:.3e}"
