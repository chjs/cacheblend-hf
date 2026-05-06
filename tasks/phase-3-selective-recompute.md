# Phase 3 — Selective KV Recompute (Core CacheBlend) ⭐

## Objective

논문 §4의 핵심 알고리즘을 구현한다. HKVD (High KV Deviation) 토큰을 식별하고, gradual filtering으로 layer마다 후보를 좁혀가며 선택적으로 KV를 재계산한다.

## Inputs

- Phase 1의 `LayerwiseModel`
- Phase 2의 `KVStore`, `apply_rope_shift`, `fuse_full_reuse`
- `docs/paper-summary.md` 의 알고리즘 섹션과 pseudo-code
- 논문 Figure 6, 7, 8 (insight 1, attention sparsity, insight 2)

## Algorithm recap

```
Input: [chunk_1, ..., chunk_N] with cached KVs (post RoPE shift applied)
       + uncached chunks (e.g., query)
Output: fused KV cache, logits

Pre: Build full concatenated input. For uncached chunks, we'll prefill normally.
     For cached chunks, we'll reuse KV and selectively recompute.

Layer 0:
    Run full prefill on layer 0 for ALL tokens (cached + uncached).
    For each cached-chunk token j, compute Δ_kv[j] = ||K_recomputed - K_cached_after_RoPE||.
    Pick top r_1% of cached tokens as candidate set S.

Layers 1..L-1:
    For tokens NOT in S: input KV from cache (with RoPE).
    For tokens IN S: recompute KV (run partial prefill on those tokens).
    Compute Δ_kv on tokens in S; narrow S to top r_l% (r_l ≤ r_{l-1}).

Layer L: final output.
```

## Steps

### 3.1 KV deviation

`src/cacheblend/hkvd.py`:

```python
def kv_deviation(
    kv_recomputed: Tensor,    # (num_kv_heads, S_subset, head_dim)
    kv_cached: Tensor,        # same shape
) -> Tensor:                  # (S_subset,)
    """L2 norm per token, averaged across heads and concatenated K+V."""
    diff = kv_recomputed - kv_cached
    # Reduce across head, dim → per-token scalar
    return diff.norm(dim=(0, 2))  # adjust based on concrete shape

def select_top_k(deviation: Tensor, k: int) -> Tensor:
    """Returns indices of top-k deviation values."""
    return deviation.topk(k).indices

def gradual_ratio_schedule(
    num_layers: int,
    target_ratio: float = 0.15,
    start_bonus: float = 0.03,
) -> list[float]:
    """
    Layer 0: not used (full prefill on all tokens).
    Layer 1: r_1 = target + bonus.
    Layers 2..L-1: linearly decay to target.
    Returns: list of length num_layers, where index 0 is None / 1.0.
    """
```

**노트**: 논문은 정확한 schedule을 명시하지 않는다. 우리의 default는 "linear decay from r_1 to r_target over first half, then flat". Phase 5에서 튜닝.

### 3.2 Partial prefill on a subset of tokens

`LayerwiseModel` 을 확장 (or wrap):

```python
def prefill_layer_partial(
    self,
    layer_idx: int,
    hidden_states: Tensor,         # (B, S_total, H) — full sequence's hidden
    position_ids: Tensor,          # (B, S_total)
    cached_kv: tuple[Tensor, Tensor],  # full-length KV (cached + recomputed for prev layer's S)
    recompute_indices: Tensor,     # (S_subset,) — indices into S_total to recompute
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    """
    1. Project hidden_states[:, recompute_indices] to get fresh Q, K, V on this layer.
    2. The 'effective' K, V used for attention is cached_kv with indices in
       recompute_indices replaced by the fresh K, V.
    3. Compute attention only for tokens in recompute_indices.
       (For all other tokens, hidden_states[:, others] is taken as-is from cache;
        we don't need to recompute their hidden output.)

    Wait — careful: hidden_states for the NEXT layer must be correct for ALL tokens
    that will need to be queried later. But if a downstream token is uncached
    (e.g., the user query), it will attend to the K of cached chunks, so we need
    correct K everywhere — which we have via the cached_kv merge.

    For the recompute_indices tokens, we DO compute their new hidden (input to next
    layer). For other tokens, their hidden remains from previous layer's
    `prefill_layer_partial` output (or from initial embedding if untouched).

    Strategy: Maintain a `hidden_states` tensor that gets updated only at
    recompute_indices on each layer. For uncached tokens (e.g., query), include
    them in recompute_indices on every layer.
    """
```

**이 부분이 Phase 3에서 가장 까다로움**. 잘못 구현하면 silent하게 wrong logits가 나온다. 검증 필수.

### 3.3 Fusor — selective recompute path

`src/cacheblend/fusor.py`:

```python
def fuse_selective(
    model: LayerwiseModel,
    chunks: list[Chunk],
    kv_store: KVStore,
    tokenizer,
    recompute_ratio: float = 0.15,
) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
    """
    Implements the algorithm above. Returns logits and the fused (= corrected) KV cache.
    """
```

### 3.4 Tests

`tests/test_selective.py`:

```python
def test_selective_better_than_full_reuse():
    """Selective recompute should produce logits closer to full recompute
    than full reuse does, on a multi-chunk input."""
    chunks = make_synthetic_chunks(n=3)
    full_logits = run_full_recompute(chunks)
    reuse_logits = run_full_reuse(chunks)
    selective_logits = run_selective(chunks, ratio=0.15)

    d_reuse = (reuse_logits - full_logits).norm()
    d_selective = (selective_logits - full_logits).norm()

    assert d_selective < d_reuse * 0.5, \
        f"Expected selective to be ≥2× closer; got reuse={d_reuse}, selective={d_selective}"

def test_selective_recovers_quality_at_prefix_position():
    """Edge case: when chunks happen to be at correct positions
    (i.e., positions match how they were precomputed), selective recompute
    should be near-perfect with very small ratio."""

def test_recompute_ratio_zero_equals_full_reuse():
    """ratio=0 should reproduce full reuse exactly."""

def test_recompute_ratio_one_equals_full_recompute():
    """ratio=1.0 should reproduce full recompute (within FP tolerance)."""

@pytest.mark.parametrize("ratio", [0.05, 0.10, 0.15, 0.20])
def test_quality_vs_ratio(ratio):
    """Logit error decreases as ratio increases (monotonic, roughly)."""
```

마지막 테스트는 monotonic decrease 를 strict하게 요구하지 말고 (sample noise) — 평균적으로 감소하는지 본다.

### 3.5 Synthetic data generator

`benchmarks/datasets/synthetic.py`:

```python
def make_synthetic_multi_chunk_input(
    num_chunks: int = 3,
    chunk_len_range: tuple[int, int] = (50, 200),
    has_cross_chunk_dependency: bool = True,
    seed: int = 42,
) -> dict:
    """
    Returns dict with 'system', 'documents', 'query', 'expected_answer'.
    For has_cross_chunk_dependency=True, the answer requires combining
    info from multiple chunks (e.g., the Messi/Ronaldo example from the paper).
    """
```

이걸로 Phase 3 테스트들이 의미 있는 케이스를 다룬다.

## Acceptance criteria

- [ ] `kv_deviation`, `select_top_k`, `gradual_ratio_schedule` 구현
- [ ] `prefill_layer_partial` 구현 및 단독 테스트 (`test_partial_equals_full_when_indices_eq_all_tokens`)
- [ ] `fuse_selective` 구현
- [ ] 위의 5가지 selective 테스트 모두 통과
- [ ] recompute_ratio=0.15 에서 logit L2 distance가 full reuse 대비 **2× 이상 작음**
- [ ] `python scripts/verify_phase.py --phase 3` 통과

## Report

- recompute_ratio 별 logit L2 distance 표 (5%, 10%, 15%, 20%, 30%, 50%)
- HKVD 분포 시각화 (Figure 7과 비슷한 plot, 가능하면)
- Insight 2 검증: layer L과 layer L+1의 HKVD top-k overlap 비율
- gradual_ratio_schedule 의 기본값과 그 근거
- Phase 4 시작 전 필요한 결정사항

## Common pitfalls

1. **hidden_states 업데이트**: layer i의 partial prefill 후, 다음 layer의 입력 hidden은 (recompute한 토큰들 + 그 외는 그대로). 이걸 헷갈리면 layer가 깊어질수록 오차가 폭증한다.
2. **Uncached tokens**: 사용자 쿼리는 KV cache가 없다. 매 layer에서 재계산되어야 한다 → `recompute_indices` 에 항상 포함.
3. **K_cached의 RoPE 적용 시점**: deviation 계산은 **post-RoPE 형태**의 K끼리 비교해야 한다. pre-RoPE끼리 비교하면 쓸모없다.
4. **Numerical stability of deviation**: L2 norm이 너무 작으면 top-k가 numerical noise를 잡는다. 합리적인 임계 또는 정규화 고려.
5. **Layer 0 vs Layer 1**: 논문 §4.3은 layer 1부터 (1-indexed) 부분 재계산을 시작한다. 우리 구현(0-indexed)에서는 layer 0이 그 역할 — 명확히 주석 처리.
