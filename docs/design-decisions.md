# Design Decisions Log

> Append-only log. Whenever a non-obvious decision is made, add an entry. Format:

```
## [YYYY-MM-DD] <Phase N> — <One-line title>

**Context**: Why this decision came up.

**Options considered**:
1. ...
2. ...

**Decision**: ...

**Reasoning**: ...

**Consequences / things to revisit**: ...
```

---

## [2025-XX-XX] <Phase 0> — Default model choice

**Context**: We need a fast-iteration model for tests and a paper-faithful model for evaluation.

**Decision**: Use `Qwen/Qwen2.5-1.5B-Instruct` for unit tests (fast); use `mistralai/Mistral-7B-Instruct-v0.2` for benchmark evaluation (matches paper).

**Consequences**: Layerwise wrapper must be model-architecture agnostic enough to handle both. Both use RoPE, so this should be fine.

---

## [2026-05-06] <Phase 1> — KV capture form: pre-RoPE primary, post-RoPE fallback

**Context**: Phase 0 reports said "store K pre-RoPE" to avoid the inverse-then-forward rotation that LMCache uses. User correction (Phase 1 prompt) softened this: try pre-RoPE first; if a single attempt at bit-exact (FP32 max_diff < 1e-5) fails, switch to post-RoPE.

**Options considered**:
1. Pre-RoPE via monkey-patching module-level `apply_rotary_pos_emb` — fragile, per-arch.
2. Pre-RoPE via custom `Attention.forward` replacement — invasive but localized.
3. Pre-RoPE via forward hook on `k_proj` — non-invasive, generic across Llama/Qwen2/Mistral (all share this submodule layout).
4. Post-RoPE only (read from `DynamicCache` after layer call) — simplest, but Phase 2 fusion must apply two rotations.

**Decision**: Option 3. `LayerwiseModel` registers a forward hook on each layer's `k_proj` that captures the K projection output and reshapes it to `(B, num_kv_heads, S, head_dim)`. This produces pre-RoPE K with **zero behavior change** to the model — the hook only observes. Post-RoPE K is also captured (from `DynamicCache.layers[i].keys`) so the wrapper can serve either form via `kv_form ∈ {"pre_rope", "post_rope"}`.

**Reasoning**:
- Hooks observe-only, so bit-exact of `model(...).logits` is not at risk.
- `k_proj` exists on every Llama-family decoder layer (Llama, Mistral, Qwen2, Qwen2.5) by the same name — so it generalizes.
- Provides both forms simultaneously, letting Phase 2 pick the cheaper rotation strategy without re-touching `LayerwiseModel`.

**Consequences / things to revisit**:
- If a future arch (Qwen3, Gemma) does extra processing between `k_proj` and `apply_rotary_pos_emb` (e.g., `k_norm` in Qwen3), the captured tensor is no longer "pure pre-RoPE" — it's "post-k_norm pre-RoPE". For correctness we'd then need the inverse k_norm to recover storage-side K. Document and handle when we add Qwen3 support.
- The `head_dim` convention (Qwen2's `head_dim = hidden_size / num_attention_heads = 1536/12 = 128`) is read from config.

---

## [2026-05-06] <Phase 1> — Use HF's standard decoder_layer + DynamicCache (no custom layer body)

**Context**: LMCache's `LMCBaseModel.compute_layer` reaches inside `vllm_model.model.layers[i].self_attn.qkv_proj` and reimplements the per-layer flow (input_layernorm → qkv → RoPE → attention → o_proj → MLP). This was tempting to mirror, but we'd need a per-arch implementation (LMCache has separate `llama.py` / `qwen3.py`).

**Decision**: `LayerwiseModel.prefill_layer` calls HF's `decoder_layer.forward(...)` with all required kwargs (`hidden_states`, `attention_mask`, `position_ids`, `past_key_values=DynamicCache`, `use_cache=True`, `cache_position`, `position_embeddings=(cos, sin)`). We do not reimplement the layer body.

**Reasoning**:
- One implementation works across Llama, Mistral, Qwen2, Qwen2.5 — they all share the decoder layer signature in transformers ≥ 4.45.
- Bit-exact with the standard forward is automatic (we are calling the exact same code).
- Forward hooks on `k_proj` give us pre-RoPE K without re-implementing the layer body.

**Consequences**: We are pinned to transformers' decoder layer signature stability. A future major version (e.g., transformers 5.x) that renames `position_embeddings` would break us. Acceptable given our pin (`transformers>=4.45,<5.0`).

---
