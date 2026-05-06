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

## [2026-05-06] <Phase 1> — CI skips model-loading tests

**Context**: Phase 1's CI run failed with `ImportError: No module named 'cacheblend'` because the workflow installed `requirements.txt` but never `pip install -e .`. Even after fixing that, every model-loading test would still try to download Qwen2.5-1.5B in CI — multi-GB, slow, and not what CI is for.

**Decision**: Tag all tests that call `from_pretrained` (or otherwise materialize a real HF model) with `@pytest.mark.requires_model`. CI runs `pytest -v -m "not gpu and not slow and not requires_model"`. A new `tests/test_smoke.py` (3 import-only tests) provides the fast packaging-regression guard CI needs.

**Reasoning**: Bit-exact correctness against a real model (Phase 1's core acceptance) belongs to local / vast.ai runs, not CI. CI's job is to catch packaging regressions early — that's what smoke tests cover.

**Consequences / things to revisit**: If we later want CI to validate against a real model, host a tiny test model on the HF Hub or check in synthetic weights and tag those tests differently.

---

## [2026-05-06] <Phase 2> — Inject cached K, V via forward hooks on k_proj / v_proj

**Context**: Phase 2's full reuse needs to run a layerwise forward where K and V at every layer come from cached chunks instead of being computed from the current hidden state. We need a way to swap in cached tensors *before* RoPE/attention without rewriting the per-arch decoder layer.

**Options considered**:
1. Replicate the layer body manually (input_layernorm → q_proj → RoPE → attention → o_proj → MLP), feeding cached K, V into attention. Per-arch code; abandons Phase 1's decision.
2. Monkey-patch `apply_rotary_pos_emb` to substitute K. Would need an inverse rotation since the cached K is pre-RoPE. Fragile.
3. **Forward hook on `k_proj` / `v_proj` whose return value replaces the projection's output**. PyTorch's `register_forward_hook` allows this. Cached pre-RoPE K is reshaped to `(B, S, kv_heads * head_dim)` (the projections' output shape) and returned by the hook; the rest of the layer (view + transpose + RoPE + attention) runs unchanged.

**Decision**: Option 3.

**Reasoning**:
- Preserves Phase 1's decision to call HF's `decoder_layer.forward(...)` unchanged.
- Generic across Llama / Qwen2 / Mistral — every Llama-family attention has `k_proj` and `v_proj` submodules.
- Pre-RoPE K is what we already store, so the hook returns the cached tensor directly; the layer's own `apply_rotary_pos_emb` then rotates it using the **fused** sequence's `position_embeddings`, which is exactly the per-chunk RoPE shift we want — no extra `apply_rope_shift` call on the hot path.
- Phase 1's capture hook on `k_proj` (which also reads `output`) is registered earlier and observes the un-overridden projection output, so it still fires correctly even with our override stacked on top. It populates `_pre_rope_k[layer_idx]` with the *fresh* K (which Phase 2 ignores).

**Consequences / things to revisit**:
- The `k_proj` and `v_proj` linears still execute and their output is discarded. For Phase 2 this is acceptable; if it ever shows up as a hot spot we'll switch to a custom layer body.
- For models with extra K processing between `k_proj` and `apply_rotary_pos_emb` (e.g., Qwen3's `k_norm`), the override drops the K-norm contribution. Document and handle when we add that arch.

---

## [2026-05-06] <Phase 2> — `apply_rope_shift` reuses model's own `rotary_emb`

**Context**: We need to RoPE-rotate a pre-RoPE K to a target position range. We could (a) reimplement the RoPE math from scratch (theta + freq table + rotate_half), or (b) ask the model's own `rotary_emb` for the (cos, sin) at the target positions and apply rotate-half in 5 lines.

**Decision**: Option (b). `apply_rope_shift(k, target_positions, model)` calls `model.compute_position_embeddings(dummy_hidden, target_positions)` (which dispatches to the model's `rotary_emb` module) and applies `(k * cos) + (rotate_half(k) * sin)`.

**Reasoning**:
- Picks up the model's RoPE config automatically: `rope_theta`, RoPE scaling (NTK / Yarn / Linear), `partial_rotary_factor` etc.
- Cannot disagree with the model on convention, dtype, or precision. Verified by `test_rope_shift_correctness` (compares against HF's own `apply_rotary_pos_emb`).

---

## [2026-05-06] <Phase 2> — KVStore disk backend deferred to Phase 4

**Context**: `tasks/phase-2-kv-storage.md` allows an optional disk backend. Phase 2's tests are entirely in-memory.

**Decision**: Implement the disk backend skeleton (`pickle.dump`/`load` in `KVStore._save_to_disk` / `_load_from_disk`), gated by `disk_dir`. Default `disk_dir=None` keeps the in-memory path. **No automated test of disk persistence in Phase 2** — Phase 4 (Pipelining) will exercise disk I/O while measuring TTFT.

**Reasoning**: Phase 2's job is correctness of full reuse + RoPE shift. The disk backend's correctness is trivial (pickle round-trip) and its only interesting property — load latency — belongs to Phase 4's pipeline tests.

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
