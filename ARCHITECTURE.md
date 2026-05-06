# Architecture

> Target module boundaries. Phase 0 ~ 5 progressively fill these in.

## High-level data flow

```
User request:
  system_prompt + [doc_1, doc_2, ..., doc_N] + user_query
                                │
                                ▼
                  ┌──────────────────────────────┐
                  │  Chunker                      │  splits input into chunks,
                  │  (src/cacheblend/chunker.py)  │  computes hashes
                  └─────────────┬────────────────┘
                                ▼
                  ┌──────────────────────────────┐
                  │  KVStore                      │  per-chunk KV lookup
                  │  (src/cacheblend/kv_store.py) │  RAM / disk backend
                  └─────────────┬────────────────┘
                                ▼ (cached + uncached chunks)
                  ┌──────────────────────────────┐
                  │  LoadingController            │  decides recompute_ratio,
                  │  (src/cacheblend/controller)  │  schedules KV prefetch
                  └─────────────┬────────────────┘
                                ▼
                  ┌──────────────────────────────┐
                  │  Fusor                        │  layer-by-layer:
                  │  (src/cacheblend/fusor.py)    │   - apply RoPE shift
                  │                               │   - select HKVD tokens
                  │                               │   - selective recompute
                  └─────────────┬────────────────┘
                                ▼
                  ┌──────────────────────────────┐
                  │  LayerwiseModel               │  HF transformers wrapper
                  │  (src/cacheblend/model.py)    │  exposes per-layer prefill
                  └─────────────┬────────────────┘
                                ▼
                              logits → next-token sampling
```

## Module responsibilities

### `model.py` — `LayerwiseModel`
Wrapper around an HF causal LM. Exposes:
- `embed_tokens(input_ids) -> hidden_states`
- `prefill_layer(layer_idx, hidden_states, position_ids, past_kv=None, attention_mask=None) -> (new_hidden, new_kv)`
- `final_norm_and_lm_head(hidden_states) -> logits`
- `num_layers`, `num_kv_heads`, `head_dim`, `rope_theta`, etc.

**No CacheBlend logic here.** This module's only job is to make the standard model callable per layer, and to verify bit-exact equivalence to `model(...)` when called sequentially.

### `kv_store.py` — `KVStore`
Hash-keyed storage of per-chunk KV tensors.
- `put(chunk_text: str, kv: List[Tensor]) -> hash`
- `get(chunk_text: str) -> Optional[List[Tensor]]`
- `evict_lru()`
- Backends: in-memory dict (default), pickled files on disk (optional).

KV stored as **position-agnostic**: K vectors are stored *before* RoPE rotation. RoPE is applied at fusion time. (Some implementations store post-RoPE and then un-rotate; we store pre-RoPE for clarity.)

### `rope.py` — RoPE utilities
- `apply_rope_shift(k: Tensor, original_pos: int, new_pos: int, rope_theta: float) -> Tensor`
- Used by the Fusor when concatenating chunks at non-prefix positions.

### `hkvd.py` — HKVD selection
- `kv_deviation(kv_recomputed, kv_cached) -> Tensor[num_tokens]`
- `select_hkvd_tokens(deviation, top_k) -> indices`
- `gradual_filter_schedule(num_layers, target_ratio) -> List[float]`

### `fusor.py` — Cache fusion
- `fuse_full_reuse(chunks, kvs) -> KVCache` — full reuse baseline (Phase 2)
- `fuse_selective(chunks, kvs, model, recompute_ratio) -> KVCache` — CacheBlend (Phase 3)

### `controller.py` — `LoadingController`
- `pick_recompute_ratio(loading_delay, prefill_delay, min_ratio=0.15) -> float`
- `schedule_prefetch(layer_idx)` — coordinates async loading of next-layer KV.

## Tensor shapes (reference)

For a model with `num_layers=L`, `num_kv_heads=H`, `head_dim=D`, sequence length `S`:

- Embedding output: `(B, S, hidden)`
- Per-layer hidden: `(B, S, hidden)`
- Per-layer K cache: `(B, H, S, D)` — same for V
- Stored chunk KV per layer: `(H, S_chunk, D)` (batch dim collapsed when stored)

## Test pyramid

```
e2e (Phase 5)             few, slow, real datasets
  ▲
pipeline tests (Phase 4)   medium, ttft measurements
  ▲
selective tests (Phase 3)  medium, deviation distribution
  ▲
fusion tests (Phase 2)     fast, synthetic chunks
  ▲
layerwise tests (Phase 1)  fast, bit-exact equivalence
```
