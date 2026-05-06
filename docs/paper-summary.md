# CacheBlend Paper — Distilled Notes

> Source: Yao et al., EuroSys 2025. This summary is operational — kept short, focused on what we need to implement.

## Problem

LLM inputs in RAG = `[chunk_1, chunk_2, ..., chunk_N, query]`. Prefill is slow and grows super-linearly with input length. We want to reuse KV cache.

- **Prefix caching**: only chunk_1's KV is reusable → marginal savings when N is large.
- **Full KV reuse (PromptCache)**: reuse all chunks' KVs. But each chunk's KV is computed independently → **cross-attention between chunks is missing** → quality drops, especially as N grows.

## Insight

Within the attention matrix, **most cross-chunk attention is sparse**: only ~10-15% of tokens have meaningfully different KV between full-recompute and cached versions. Recomputing just those tokens (HKVD = High KV Deviation tokens) recovers most of the quality.

Two empirical insights drive the algorithm:

- **Insight 1**: Recomputing KV of tokens with higher KV deviation (Δ_kv) reduces attention deviation (Δ_attn) more.
- **Insight 2**: HKVD tokens are highly correlated across consecutive layers (Spearman correlation visible in Fig. 8 of paper). So we don't need to know HKVD for layer L+1 a priori — we can pick from layer L's candidates.

## Algorithm (the bit we implement)

For an LLM input split into chunks `c_1, ..., c_N`, with pre-computed KV for each chunk:

```
Layer 1:
    Run full prefill on layer 1 for ALL tokens
    Compute Δ_kv[j] = ||KV_1_recomputed[j] - KV_1_cached[j]||  for each token j
    Pick top r1% indices as HKVD candidates → S_1

Layer i (i ≥ 2):
    For tokens in S_{i-1} only, recompute KV at layer i
    Compute Δ_kv[j] for j in S_{i-1}
    Pick top r_i% (r_i ≤ r_{i-1}) → S_i
    For tokens NOT in S_{i-1}: reuse cached KV (with RoPE position shift if needed)
```

Eventually most layers recompute only ~15% of tokens. Total compute ≈ 15% of full prefill.

### Position recovery (RoPE)

A chunk cached at position p1 needs to be reused at position p2. With RoPE, this is a single per-vector rotation:

```
K[m+l] = R_Θ,m+l · K       where R is the RoPE rotation matrix
```

Because RoPE attention scores depend only on **relative** position (Appendix A of the paper), shifting both K and Q by the same delta is equivalent. In practice we shift only K of cached chunks to land them at their new absolute position.

### Recompute ratio schedule (gradual filtering)

The paper isn't fully prescriptive here. Reasonable defaults:
- `r_1 = target_ratio + 3%` (slightly larger candidate set on layer 1)
- `r_i = max(target_ratio - 1%, target_ratio)` decaying over layers
- For our implementation: linearly decay from `r_1` to `r_target` over the first ~half of layers, then keep flat.

We will treat this as a hyperparameter and tune in Phase 5.

## Empirical figures we depend on

### Figure 6 — recompute ratio vs forward attention deviation
The paper sweeps `r ∈ {0%, 5%, 10%, 15%, 18%, …}` and plots the L2 distance between the *forward attention matrix* (last-token rows) under selective recompute vs full recompute. The curve is sharply convex: the first ~15% of recomputed tokens removes most of the deviation, after which gains flatten. **Operational implication for us**: 15% is not a tuning artifact; it is the elbow of the curve. Below ~10% quality drops noticeably; above ~20% we pay compute for little gain. We default `target_ratio = 0.15` and only revisit it if Phase 5 shows F1 outside the 0.02 budget.

### Figure 8 — HKVD rank correlation across layers
For each layer ℓ, Spearman correlation of the per-token KV deviation with that of layer ℓ−1 is consistently high (≳0.7) once past the first 1–2 layers. **Operational implication**: gradual filtering is sound — at layer ℓ we may restrict the candidate set to ≤ |S_{ℓ-1}| without losing meaningful HKVD coverage. The first one or two "check layers" must compute deviation over **all** reused tokens (a full pass) so that S_1 is set from the actual top-k; subsequent layers prune within S_{ℓ-1}. LMCache's `blend_check_layers` config exposes exactly this knob (default: only layer 1 runs the check).

## §5 — LoadingController (algorithm we partially implement)

The controller adaptively picks `recompute_ratio r` so that **selective-recompute latency ≤ KV-load latency**, hiding loading entirely.

Notation:
- `T_recompute(r)` = wall time for selective recompute at ratio `r` per layer. Roughly linear in `r` for fixed model/batch: `T_recompute(r) ≈ a·r + b` with `b` the fixed overhead (RoPE, scatter, mask building).
- `T_load` = wall time to load one layer of KV from the chosen storage tier (RAM ~ μs/layer, NVMe ~ tens of ms/layer, network slower).
- `r_min` = quality floor (paper uses 0.15; below this F1 falls outside 0.02).
- `r_max` = ratio at which selective recompute equals full recompute (≈1.0 minus eviction overhead).

```
choose_ratio(T_load, model_profile):
    # smallest r such that T_recompute(r) >= T_load
    r_match = (T_load - b) / a
    return clamp(r_match, r_min, r_max)
```

A second decision is **storage tier**: pick the slowest tier whose `T_load` still satisfies `T_recompute(r_min) ≥ T_load` — this maximises capacity at no latency cost. If no tier matches, fall back to the next-faster tier and re-run `choose_ratio`.

For Phase 4 we implement a simplified `LoadingController.pick_recompute_ratio(loading_delay, prefill_delay, min_ratio=0.15)` that linearly interpolates between `r_min` and `r_max` based on the measured load/recompute ratio; multi-tier auto-selection is **deferred** unless Phase 5 motivates it.

## System (the bit we partially implement)

- KV stored on disk (NVMe SSD in paper) and loaded in parallel with selective recompute → no extra TTFT.
- LoadingController picks recompute ratio so that recompute_delay ≤ load_delay.
- We implement a simplified version in Phase 4. Multi-tier storage etc. is out of scope.

## Numbers from the paper (sanity targets)

- Mistral-7B, 4K context: layer-wise selective recompute (15%) ≈ 3 ms/layer; SSD load ≈ 16 ms/layer → loading dominates.
- Llama-70B, 4K context: 15% recompute ≈ 7 ms/layer; SSD load ≈ 4 ms/layer → recompute dominates.
- F1 drop vs full recompute: ≤ 0.02 across datasets at 15-18% recompute.
- TTFT speedup vs full recompute: 2.2–3.3×.

## What we deliberately skip

- vLLM integration (we use HF transformers directly)
- Multi-tier (RAM + SSD + cold) storage selection (we do single-tier)
- Quantization
- Eviction beyond simple LRU
- Multi-GPU / tensor-parallel concerns

## Pseudo-code we will implement (Phase 3)

```python
def cacheblend_prefill(model, input_chunks, cached_kvs, recompute_ratio):
    # 1. Embed all tokens
    hidden = model.embed_tokens(concat(input_chunks))
    pos_ids = compute_position_ids(input_chunks)

    # 2. Apply RoPE position shift to cached K's
    cached_kvs = [rope_shift(kv, target_pos) for kv, target_pos in cached_kvs]

    # 3. Layer 0: full prefill, compute deviations, pick HKVD candidates
    new_hidden, fresh_kv = model.prefill_layer(0, hidden, pos_ids)
    deviation = ||fresh_kv - cached_kvs[0]||  # only over reused positions
    candidates = topk(deviation, r1 * num_reused_tokens)

    # 4. Layers 1..L-1: selective recompute
    for layer in range(1, num_layers):
        # candidates' KV is recomputed; everyone else reuses cached
        kv_for_layer = scatter(cached_kvs[layer], candidates, recomputed_kv)
        new_hidden, recomputed_kv_at_candidates = model.prefill_layer_partial(
            layer, new_hidden, pos_ids, kv_for_layer, mask=candidates
        )
        # narrow candidates further
        deviation_at_cands = ||recomputed_kv_at_candidates - cached_kvs[layer][candidates]||
        candidates = candidates[topk(deviation_at_cands, r_{layer} * |candidates|)]

    # 5. Final norm, lm_head
    return model.final_norm_and_lm_head(new_hidden)
```

This is the pseudo-code; the real implementation needs to handle the new (uncached) tokens — typically the user query and any uncached chunks — with full prefill alongside the cached chunks.
