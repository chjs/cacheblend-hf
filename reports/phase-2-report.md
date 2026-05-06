# Phase 2 Report — KV Storage & Full Reuse with RoPE Recovery

## Summary

청크 단위 pre-RoPE KV 저장소(`KVStore`), `apply_rope_shift`(model의 `rotary_emb` 재사용), `precompute_chunk_kv`, `fuse_full_reuse`를 구현. 단일 prefix 청크 케이스에서 `full_reuse_logits`가 `full_recompute_logits`와 정확히 일치(`max_diff = 0.000e+00`); 다중 청크 케이스에서는 두 번째 청크 위치에 발산이 100% 집중(`L2 = 1239, max_diff = 7.21`)되어 논문이 기록한 cross-attention 누락이 그대로 재현됨. Phase 3에서 selective recompute로 좁힐 base error budget이 명확해졌다.

## What was done

### Implemented

- `src/cacheblend/chunker.py` — `Chunk` dataclass + `chunk_texts(texts, tokenizer)`. 해시는 텍스트 SHA-256 16자(위치 무관). 토큰화는 `add_special_tokens=False`, fused 입력은 chunk token_ids의 `torch.cat`로 — concat-tokenize 차이 회피.
- `src/cacheblend/kv_store.py` — `KVStore` (in-memory `OrderedDict` LRU + opt-in pickled-file backend). 저장: `list[(K_pre_rope, V)]` of length `num_layers`, K shape `(1, num_kv_heads, S, head_dim)`. 디스크 backend는 path 인자로 켜지며, Phase 4까지 자동 테스트 X.
- `src/cacheblend/rope.py` — `apply_rope_shift(k_pre_rope, target_positions, model)`. model의 `compute_position_embeddings`로 (cos, sin) 받아 `(k * cos) + (rotate_half(k) * sin)` 인라인. 별도 RoPE 재구현 없음 → rope_theta/scaling/dtype 자동 일치.
- `src/cacheblend/precompute.py` — `precompute_chunk_kv(model, text, tokenizer)` + `precompute_chunk_kv_from_ids(model, ids)`. `LayerwiseModel.prefill_layer`를 layer 0..L-1 순회하며 per-layer (K, V) 수집.
- `src/cacheblend/fusor.py::fuse_full_reuse(model, chunks, kv_store)` — layer 루프 내에서 `k_proj`/`v_proj`에 forward hook을 일시 등록해 cached pre-RoPE K/V를 layer 출력으로 주입. HF의 attention이 fused-sequence position_embeddings로 RoPE를 자동 적용 → 명시적 `apply_rope_shift` 호출 hot-path에서 제거. Phase 1의 "decoder_layer 직접 호출" 결정 보존.

### Tests (`tests/test_kv_reuse.py`)

| Test | Result | Numbers |
|---|---|---|
| `test_rope_shift_correctness` | ✅ pass | max_diff = **0.000e+00** vs HF `apply_rotary_pos_emb` |
| `test_full_reuse_matches_full_recompute_when_only_one_chunk_at_prefix` | ✅ pass | max_diff = **0.000e+00**, S=4 |
| `test_full_reuse_diverges_with_multiple_chunks` | ✅ pass | max_diff = **7.208**, L2 (전체) = **1.239e+03**, L2 (chunk 2 단독) = **1.239e+03**, S=20 |

`pytest -v -m "requires_model and not slow"` (Phase 1 회귀 포함, 5 tests): **5 passed in 14:48**.
`pytest -v -m "not gpu and not slow and not requires_model"` (CI selection): **6 passed in 4s**.

### Acceptance criteria checklist

- [x] `Chunker`, `KVStore`, `apply_rope_shift`, `precompute_chunk_kv`, `fuse_full_reuse` 모두 구현
- [x] `test_rope_shift_correctness` 통과 (max_diff = 0)
- [x] `test_full_reuse_matches_full_recompute_when_only_one_chunk_at_prefix` 통과 (max_diff = 0)
- [x] `test_full_reuse_diverges_with_multiple_chunks` 통과 (L2 = 1239, divergence가 chunk 2에 100% 집중)
- [x] `python scripts/verify_phase.py --phase 2` 통과

## RoPE shift 구현 방법

**model의 `rotary_emb` 재사용**. `apply_rope_shift`는 자체적으로 RoPE 식을 다시 쓰지 않고:

```
dummy_hidden = zeros(B, S, hidden_size, dtype=k.dtype)
cos, sin = model.compute_position_embeddings(dummy_hidden, target_positions)  # → model.rotary_emb 호출
return (k * cos.unsqueeze(1)) + (rotate_half(k) * sin.unsqueeze(1))
```

`rotate_half`만 5라인으로 인라인 (`x1, x2 = x.chunk(2, dim=-1); cat([-x2, x1], -1)`). HF Llama-family의 `apply_rotary_pos_emb`와 비교해 max_diff = 0.000e+00 (test_rope_shift_correctness). `rope_theta`/RoPE scaling/`partial_rotary_factor` 등 모든 RoPE 메타정보는 model의 `rotary_emb` 모듈이 들고 있으므로 자동 적용.

## KVStore disk backend 여부

skeleton은 구현 (`pickle.dump`/`load`, `disk_dir` 인자로 활성화), default는 in-memory only. Phase 2의 자동 테스트는 in-memory만 사용. 디스크 backend의 의미 있는 검증은 latency 측정이 본질이므로 **Phase 4 (Pipelining)에서 TTFT 측정과 함께 진행**한다고 design-decisions에 기록.

## Divergence test의 logit L2 distance 수치

`test_full_reuse_diverges_with_multiple_chunks` (S=20, 두 청크):

| Metric | Value |
|---|---|
| max_diff | 7.208e+00 |
| L2 (전체 시퀀스) | 1.239e+03 |
| L2 (chunk 1 위치, 첫 8 토큰) | ~0 (single-chunk-at-prefix와 동일 경로) |
| L2 (chunk 2 위치, 마지막 12 토큰) | 1.239e+03 |

발산이 두 번째 청크에 **100% 집중**됨 — 첫 청크는 prefix(position 0)에 있어 cached K/V가 fresh K/V와 동일하므로 logit 차이 없음. 두 번째 청크는 cross-attention 부재로 hidden flow가 다르고, 그 결과 cached K/V vs fresh K/V 차이가 누적됨. 이는 논문 §3.3의 PromptCache/full reuse 실패 양상 그대로.

Phase 3에서 selective recompute로 이 L2를 ~ 1e-1 수준 또는 그 이하로 줄이는 것이 목표 (논문의 forward attention deviation 곡선 elbow ≈ 15% 토큰 recompute에서).

## Phase 1 회귀 (long-test max_diff 측정)

`pytest tests/test_layerwise.py::test_layerwise_matches_standard_longer -v -s -m "slow and requires_model"` (S=63):

```
[bit-exact long] max_diff = 0.000e+00, S=63
1 passed in 164.57s (0:02:44)
```

**Phase 1 fast (S=5)와 long (S=63) 양쪽 모두 max_diff = 0.000e+00.** Layerwise wrapper의 base error는 측정 한계 이하 (FP32 CPU에서 모든 비교가 exact bit-identical). Phase 2~3의 error budget의 baseline이 사실상 0이므로, Phase 3 selective recompute 후의 logit L2가 그대로 "CacheBlend가 도입한 추가 오차" 수치로 해석 가능.

## Decisions made

(전체는 `docs/design-decisions.md` 참조)

- **forward hook on `k_proj`/`v_proj`**로 cached K, V를 layer attention에 주입 — Phase 1의 "decoder_layer 직접 호출" 결정 보존하면서 K/V swap 가능. 1차 시도 성공.
- **`apply_rope_shift`는 model의 `rotary_emb` 재사용** — RoPE 식 재구현 회피, model 설정 자동 일치.
- **KVStore disk backend는 Phase 4로 미룸** — Phase 2의 본질은 정확성, 디스크 latency는 Phase 4 TTFT 측정과 함께.

## Deviations from plan

- `tasks/phase-2-kv-storage.md`의 `chunk_input(system_prompt, documents, user_query, tokenizer)` 시그니처를 일반화한 `chunk_texts(texts, tokenizer)`로 구현. system_prompt/documents/user_query는 caller가 list로 만들어 넘김. 이유: Phase 2 테스트는 단일 청크 / 다중 청크 두 케이스만 필요해서 RAG 특화 시그니처가 과한 가정.
- `precompute_chunk_kv`의 `use_dummy_prefix_for_position` 인자 미구현 (사용처 없음). 우리는 RoPE 회전을 직접 적용하므로 dummy prefix는 필요 없다는 phase 0 결정과 일치.
- `fuse_full_reuse`의 cache miss는 raise (현재). Phase 4에서 prefetch 또는 on-demand precompute로 보완.

## Open questions / blockers

1. **Tokenizer 일관성과 BPE 경계**: 현재 chunker는 각 청크 텍스트를 `add_special_tokens=False`로 독립 토큰화. 일반적으로 BPE 토크나이저는 두 텍스트를 concat한 토큰화와 각각 토큰화한 후 concat한 결과가 다를 수 있음(공백/특수문자 경계). 우리 테스트의 짧은 영어 문장에서는 문제없었으나, Phase 5 벤치마크 전에 실제 데이터셋에서 mismatch 빈도와 영향 측정 필요.
2. **Qwen3 등 k_norm 있는 arch**: 현재 hook은 `k_proj`의 출력을 cached pre-RoPE K로 override함. Qwen3는 `k_proj` 후 `k_norm`이 추가되므로 hook의 semantics가 깨짐. 현재는 Qwen2/Llama/Mistral 한정. 필요 시 `k_norm` 직전이 아닌 직후에 hook을 거는 옵션 추가.

## Files changed

```
docs/design-decisions.md              | +60 (Phase 2 결정 3건)
docs/prompts/phase-2-kv-storage.md    | +61 (신규, 본 phase 프롬프트)
reports/phase-2-report.md             | +N (이 파일)
src/cacheblend/chunker.py             | +57 (Chunk dataclass + chunk_texts)
src/cacheblend/fusor.py               | +110 (fuse_full_reuse via k_proj/v_proj override)
src/cacheblend/kv_store.py            | +88 (in-memory + opt-in disk)
src/cacheblend/precompute.py          | +60 (precompute_chunk_kv + _from_ids)
src/cacheblend/rope.py                | +60 (apply_rope_shift)
tests/test_kv_reuse.py                | +110 (3 phase-2 tests)
```

## Next phase prep (Phase 3 — Selective KV Recompute)

- `fuse_full_reuse`의 layer 루프가 그대로 selective recompute의 뼈대. 차이는: hook이 cached K, V를 그대로 주입하는 대신 일부 토큰 위치는 fresh K, V로 교체.
- HKVD selection은 layer 1 (default check layer)에서 fresh K_layer1과 cached K_layer1의 squared L2 차이를 토큰별로 계산해 top 15% 인덱스 추출.
- LMCache처럼 단일 ratio·단일 check layer로 시작 (per-layer decay schedule은 Phase 5 F1 측정 후 필요시 도입).
- Phase 2의 divergence baseline (L2 = 1239, 두 청크/S=20)이 그대로 Phase 3의 "줄여야 할 양". 같은 입력에서 selective recompute 후 L2 < 100 정도면 elbow 진입, < 10이면 본격 복원.

## GitHub PR

PR URL: **https://github.com/chjs/cacheblend-hf/pull/2**

Branch: `phase-2-kv-storage` → `main`. CI 녹색 확인 후 사용자 머지 권장.

## Suggested next prompt for Claude Code

> Phase 2 PR을 main으로 머지한 뒤 Phase 3을 진행하세요.
>
> 다음 파일을 먼저 읽으세요: `tasks/phase-3-selective-recompute.md`, `docs/paper-summary.md`(Insight 1/2, gradual filtering), `docs/design-decisions.md` (Phase 2의 hook 기반 K/V swap 메커니즘).
>
> Phase 2 상속 컨텍스트:
> - `fuse_full_reuse`의 layer 루프 + `k_proj`/`v_proj` forward hook이 selective recompute의 뼈대. layer 1에서 fresh K vs cached K 차이로 HKVD 토큰 선정 후, 그 토큰만 fresh K/V로 두고 나머지는 cached로 두는 hook을 등록.
> - Phase 2 divergence baseline: 2개 청크 S=20에서 L2 = 1239. Phase 3 목표: ratio=15%에서 L2 < 100 (elbow 진입), 가능하면 < 10.
> - Phase 1 base error는 max_diff = 0 (FP32 CPU). Phase 3의 모든 logit 차이는 100% selective recompute가 만든 차이로 해석됨.
> - check layer는 LMCache처럼 단일(layer 1)부터. per-layer ratio decay는 미구현 — Phase 5 F1 측정 후 필요시.

---

Prompt archive: `docs/prompts/phase-2-kv-storage.md`
