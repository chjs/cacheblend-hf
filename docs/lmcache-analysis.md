# LMCache CacheBlend — Analysis

> **Status**: Filled in Phase 0.
>
> **Source**: `external/LMCache` cloned from `github.com/chjs/LMCache` branch `fix/cacheblend-vllm-v0.17.1-compat` (commit `9f8aa4d`, merge of `dev` into the compat branch).
>
> The goal: identify exactly what to keep (in spirit) and what to drop when porting to plain HF Transformers. Read at *module* granularity — not line-by-line.

---

## Top-level orientation

LMCache is a vLLM plugin. Its CacheBlend implementation lives almost entirely under `lmcache/v1/compute/` and is wired into the per-request prefill via the vLLM-side adapter. The relevant tree:

```
lmcache/v1/compute/
├── blend/
│   ├── blender.py            (LMCBlender — selective recompute + HKVD selection)
│   ├── metadata.py           (LMCBlendCommonMetadata, LMCBlendMetadata)
│   ├── utils.py              (LMCBlenderBuilder — singleton-per-engine factory)
│   └── __init__.py
├── models/
│   ├── base.py               (LMCBaseModel — layerwise prefill driver, @torch.compile)
│   ├── llama.py              (Llama/Qwen2/Mistral _process_qkv — passthrough)
│   ├── qwen3.py              (q_norm/k_norm pre-RoPE)
│   └── utils.py              (infer_model_from_vllm dispatcher; VLLMModelTracker)
├── attention/
│   ├── flash_attn.py         (LMCFlashAttnBackend, dense varlen path)
│   ├── flash_infer_sparse.py (sparse block-sparse attention via flashinfer)
│   ├── metadata.py           (LMCFlashAttnMetadata, LMCFlashInferSparseMetadata)
│   └── abstract.py           (AttentionInterface)
└── positional_encoding.py    (BasicReverseRope, FusedRope, get_fused_rope)
```

The engine glue (`lmcache/v1/cache_engine.py`, `lmcache/v1/token_database.py`, `lmcache/integration/vllm/vllm_v1_adapter.py`) is **out of scope** for our port — it deals with vLLM page tables, multi-tier storage, and request scheduling.

---

## Files / dirs read

### `lmcache/v1/compute/blend/` — the CacheBlend logic

- **What's here**: `LMCBlender` orchestrates per-layer selective recompute. It owns a `layerwise_model` (driver of vLLM's stacked decoder layers via a generator), receives per-layer `(q, k, v, residual)` from that driver, applies RoPE on Q and the freshly-computed K, computes per-token KV deviation against the cached K (`old_k`), top-k selects HKVD indices once at a configured "check layer", scatters the recomputed K/V into the cached K/V at those indices, and returns the masked tensors so attention is computed only over the HKVD slice.
- **What we'll keep (in spirit)**: the *control flow* — layerwise generator, in-place scatter of recomputed KV into cached KV, deviation-then-topk in a single check layer.
- **What we'll drop**: vLLM coupling (`vllm_model.model.layers[i].self_attn.qkv_proj` etc.), the singleton `LMCBlenderBuilder` registry, the sparse FlashInfer fast path, the per-layer `attn_metadata.update_from_top_indices` plumbing (we'll just rebuild masks from indices each layer).

### `lmcache/v1/cache_engine.py` — main engine entry point

- **Key observation**: blending is enabled by `LMCacheEngineConfig.enable_blending`. When set, the engine (a) chooses `MemoryFormat.KV_2TD` (keeps K and V separately addressable per token, layer-by-layer), and (b) instantiates `SegmentTokenDatabase` instead of the default `ChunkedTokenDatabase` so chunk boundaries are derived from a literal separator string (`blend_special_str`, default `" # # "`) tokenized into `sep_tokens` and matched via a sliding-window `unfold` in `_fast_split_by_subtensor`. Each chunk is hashed by its own tokens (position-independent) → enables non-prefix reuse.
- **How it dispatches to blend**: not from the engine itself. The vLLM adapter (`vllm_v1_adapter.py:817`) calls `self.blender.blend(tokens[:lmcache_cached_tokens], token_mask, kvcaches=…, slot_mapping=…)` after the load step decides which tokens are cached.

### HKVD selection logic

- **File**: `lmcache/v1/compute/blend/blender.py`
- **Key function**: `LMCBlender.process_qkv(...)`, lines 88–113.
- **Notes**: The selection happens inside `if layer_id in self.common_metadata.check_layers:` — by default `check_layers=[1]` (only layer index 1, after layer 0's full pass). `diff_k = sum((k_fresh - k_cached)^2, dim=hidden)` — i.e., **squared L2 over the head/hidden dim**, *not* an absolute or relative norm; per token. Top-k is `int(total_len * recomp_ratios[0])` (a single ratio, not per-layer; the field is a list but only `[0]` is used — there's a `TODO(Jiayi)` to support per-layer ratios). `top_indices` are then **sorted** so downstream causal masking still works; `imp_indices` is stored on `metadata` so subsequent layers reuse the same slice (this is gradual filtering's degenerate form: the same set across all layers ≥ check layer, no further narrowing).
- **Divergence from paper**: paper's `r_1 > r_2 > … > r_target` decay is **not** implemented; LMCache uses one constant ratio applied at the check layer and never re-narrows.

### RoPE position recovery

- **File**: `lmcache/v1/compute/positional_encoding.py`
- **Key functions**:
  - `BasicReverseRope.reverse_encode(positions, q, k)` — pure-PyTorch reverse rotation: shuffle (neox vs interleaved layouts handled separately), call vLLM's forward `rope`, shuffle back. Used for correctness validation, not the hot path.
  - `FusedRope.fused_encode(old_positions, new_positions, k)` — production hot path. Calls a CUDA fused kernel `lmc_ops.rotary_embedding_k_fused` that, in one pass, applies `R^{-1}_{old}` then `R_{new}` to K — i.e., **rotates K from its cached position to its new absolute position** without going through float-domain reverse rotation in Python.
  - `get_fused_rope(...)` — factory; validates `rotary_dim == head_size`, `rope_scaling is None`, `partial_rotary_factor == 1.0`. **If any of these don't hold, blending is disabled.** This is a real constraint we inherit.
- **Notes**: K is stored *post-RoPE at original position* (i.e., already rotated by `R_{cached_pos}`); fusion-time op rotates by `R_{new_pos} · R^{-1}_{cached_pos}`. (Our `ARCHITECTURE.md` chose to store **pre-RoPE** instead — see §"Decisions log" at bottom.)

### KV storage unit — token? chunk? page?

- **Memory format**: `MemoryFormat.KV_2TD` when blending is on (vs `KV_T2D` for standard layerwise, vs `KV_MLA_FMT` for MLA models). `KV_2TD` is a layout where each per-layer object packs `(K, V)` separated, then **token-major then dim** — i.e., per-layer storage of shape `(2, T, D)` where T is the chunk's token count and D is the flattened head*head_dim. The store+retrieve interface is **per token within a per-layer per-chunk object**. The engine's coarsest unit is the chunk (variable length, demarcated by `blend_special_str`), but inside a chunk individual tokens are addressable for the scatter-by-`imp_indices` step.
- **Eviction / tiers**: `MixedMemoryAllocator` with CPU + optional disk + optional remote (via `storage_backend/`). Our port keeps only an in-memory dict with optional pickled disk fallback; multi-tier and async allocator out of scope.

### Recompute ratio scheduling

- **Where**: `LMCBlender.__init__` reads `config.blend_check_layers` (which layers run the deviation+topk pass) and `config.blend_recompute_ratios` (a list, but only index `[0]` is consumed in `process_qkv`).
- **How layer-per-layer ratio is decided**: it isn't, in this fork. There's a `TODO(Jiayi): support different ratios for different layers` and another `TODO` to remove the `[0]` hardcode. So LMCache currently runs with a single ratio applied at one (or a few) check layers; the paper's gradual decay is left as future work.
- **Implication for us**: our `hkvd.gradual_filter_schedule(num_layers, target_ratio)` is **net-new** vs LMCache. We have to design this ourselves; LMCache won't be a reference. The simplest defensible default is the schedule already in our `paper-summary.md` (linear decay from `r_target+0.03` to `r_target` over the first half of layers, then flat).

### vLLM integration glue

**Out of scope.** Boundaries to avoid:

- `lmcache/integration/vllm/vllm_v1_adapter.py` — request lifecycle, slot mapping, `kvcaches` tensor handle injection. We will instead use HF's `past_key_values` / our own `KVStore`.
- `lmcache/v1/compute/models/base.py` (lines 73–142) — `compute_layer` is a generator that drives `vllm_model.model.layers[…]` directly: input layernorm → `layer.self_attn.qkv_proj` → `_process_qkv` → `blender.process_qkv` → `lmc_attn_layers[idx].forward_contiguous` → `o_proj` → MLP. We replicate this *shape* but on HF transformers' `<arch>DecoderLayer` instead, which exposes a clean `forward(hidden_states, attention_mask, position_ids, past_key_value, ...)`.
- `lmcache/v1/compute/attention/flash_attn.py`, `flash_infer_sparse.py` — vLLM's `FlashAttentionImpl` and FlashInfer wrappers. We use HF's eager / SDPA / FA2 paths via `attn_implementation=...`.

### Tests

- LMCache has `tests/v1/multiprocess/test_blend_server_v2.py` (the multiprocess blend server) — **not** unit tests of the blender or HKVD logic itself. There's no `test_blend.py` covering the core math we're porting. So our `tests/test_kv_reuse.py` and `tests/test_selective.py` will need to be designed from first principles + the paper, not adapted from theirs.

---

## Mapping: LMCache → our repo

| LMCache concept | Our equivalent | Notes |
|---|---|---|
| `LMCBlender.process_qkv` | `fusor.fuse_selective` (per-layer body) | We split: deviation+topk lives in `hkvd.py`, scatter+attention call stays in `fusor.py`. |
| `LMCBlender.blend_layer` (generator) | `model.LayerwiseModel.prefill_layer(...)` driven by `Fusor` loop | We don't need a generator — a plain `for layer_idx in range(L)` loop suffices since we don't pipeline retrieve+compute in Phase 1–3. |
| `LMCBaseModel.compute_layer` | `LayerwiseModel.prefill_layer` + `embed_tokens` + `final_norm_and_lm_head` | HF's `<Arch>DecoderLayer.forward` already encapsulates qkv_proj→attn→o_proj→mlp; we don't reimplement that. Phase 1's bit-exact test guards against any divergence. |
| `LMCBlendCommonMetadata` | constructor args of `Fusor`/`HKVDSelector` | No need for a dataclass; small enough. |
| `BasicReverseRope` / `FusedRope` | `rope.apply_rope_shift(k, original_pos, new_pos, rope_theta)` | Pure-PyTorch implementation; no CUDA op. We store **pre-RoPE**, so we apply `R_{new}` once at fuse time (no inverse). |
| `SegmentTokenDatabase` (separator-based chunking) | `chunker.py` | We expose chunks explicitly via the API (caller passes `List[str]`); no separator hack. |
| `KV_2TD` memory format + `MixedMemoryAllocator` | `KVStore` in-memory dict (+ optional pickled-file backend) | Single tier in Phase 2; multi-tier out of scope. |
| `blend_recompute_ratios` (scalar in practice) | `Controller.pick_recompute_ratio` + `hkvd.gradual_filter_schedule` | Net-new logic on our side. |
| `attention/flash_infer_sparse.py` | (none) | Sparse-attention fast path — out of scope. |
| `vllm_v1_adapter.py` | (none) | vLLM-only. |

---

## What's most threatening to our simplicity

1. **vLLM coupling, even inside `compute/`**: `LMCBaseModel.compute_layer` reaches into `layer.self_attn.qkv_proj`, `q_size/kv_size`, and `layer.self_attn.rotary_emb` directly. Tempting to mirror — but HF transformers exposes these via `LlamaDecoderLayer.self_attn` differently per arch. We must resist the urge to also write per-arch model adapters. Instead, call `decoder_layer(hidden_states, position_ids=..., past_key_value=..., use_cache=True)` and let HF do the qkv split internally; intercept the returned `past_key_value` to compute deviation. Plan to verify this works in Phase 1 with the bit-exact test.

2. **RoPE storage convention**: LMCache stores **post-RoPE K** and applies a fused inverse-then-forward rotation. Their fused kernel exists *because* the post-RoPE convention requires two rotations per fusion. We should store **pre-RoPE K** and rotate **once** at fusion time. This is simpler, has no kernel dependency, and matches the paper's appendix more directly. **Cost**: standard HF caches store post-RoPE K; we either run our own `LayerwiseModel.prefill_layer` to capture K *before* `apply_rotary_pos_emb`, or we compose-back from post-RoPE by storing the original positions alongside. The former is cleaner — log this decision.

3. **`@torch.compile` on `compute_layer`**: LMCache compiles the per-layer body. We should **not** compile in Phase 1–3 — it complicates debugging the bit-exact test and adds first-call latency that pollutes any timing we do. Only consider for Phase 4 timing measurements, and only if needed.

4. **Single recompute ratio across layers**: LMCache punts on the gradual decay schedule. We have to design it from the paper alone, and it's the most likely place to spend hours tweaking. **Mitigation**: hardcode a sensible default, defer tuning to Phase 5, do not let Phase 3 block on this.

5. **Sparse attention path** (`flash_infer_sparse.py`): tempting to think we'll need block-sparse attention to make selective recompute fast. We will not, because Phase 1–4 tests are correctness/relative speed, not absolute throughput. Use HF's standard SDPA over the masked slice. If absolute TTFT in Phase 4 forces us to revisit, document the gap explicitly.

---

## Decisions log (Phase 0)

- **2026-05-06**: Store K **pre-RoPE** (our convention) vs LMCache's **post-RoPE**. Reason: avoids the inverse-rotation step and removes the need for a fused CUDA kernel. Cost: must intercept K before `apply_rotary_pos_emb` inside `LayerwiseModel.prefill_layer`. Revisit only if a Phase 1 bit-exact test fails because we can't cleanly split that step in some HF arch.
- **2026-05-06**: Do not implement per-layer recompute ratio decay in Phase 3. Use a single ratio (paper's 15%) at all layers ≥ check layer, matching LMCache's actual behavior. Add `gradual_filter_schedule` only if Phase 5 F1 exceeds the 0.02 budget.
- **2026-05-06**: Single check layer (layer index 1), as LMCache's default. Configurable but we do not tune it.
- **2026-05-06**: No `@torch.compile` until at least Phase 4. Bit-exact testing comes first.
- **2026-05-06**: No sparse attention backend. HF's standard attention path over a masked slice is sufficient through Phase 4.
