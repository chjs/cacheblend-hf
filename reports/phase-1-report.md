# Phase 1 Report — Layerwise Forward

## Summary

`LayerwiseModel`을 구현해 HF transformers의 stacked decoder layer를 layer 단위로 호출 가능하게 했고, Qwen2.5-1.5B-Instruct (FP32, CPU)에서 **표준 forward와 logit이 정확히 동일** (`max_diff = 0.000e+00`)하게 일치함을 확인. KV는 pre-RoPE / post-RoPE 두 form 모두 지원하며, 1차 시도(pre-RoPE via `k_proj` forward hook)에서 bit-exact 테스트가 통과해 fallback이 불필요했다.

## What was done

### Implemented

- `src/cacheblend/model.py` — `LayerwiseModel` 클래스 296라인.
  - `embed_tokens(input_ids) -> hidden`
  - `compute_position_embeddings(hidden, position_ids) -> (cos, sin)` — model의 `rotary_emb`을 한 번 호출해 layer 간 공유.
  - `build_causal_mask(...)` — `transformers.masking_utils.create_causal_mask`를 그대로 사용.
  - `prefill_layer(layer_idx, hidden, position_ids, position_embeddings, ...) -> LayerOutput(hidden, k, v)` — HF의 `decoder_layer.forward`를 정확한 시그니처로 호출 (`use_cache=True` + `DynamicCache`).
  - `final_norm_and_lm_head(hidden) -> logits`
  - `forward_layerwise(input_ids) -> logits` — 위 메서드들을 묶어 한 번에 logits 계산 (테스트 편의용).
  - 내부: `k_proj`에 forward hook을 걸어 pre-RoPE K를 캡처. `kv_form ∈ {"pre_rope", "post_rope"}`로 어느 form을 반환할지 선택.

- `tests/test_layerwise.py` — 3개 테스트 (총 8:38).
  - `test_layerwise_matches_standard` — `model(...).logits` vs `LayerwiseModel.forward_layerwise(...)`, FP32 CPU, 5-token 입력, `max_diff < 1e-5` 단언.
  - `test_layerwise_matches_standard_longer` — 동일, ~75 tokens 입력. `@pytest.mark.slow` (CPU FP32에서 5분+).
  - `test_kv_extraction` — `kv_form="post_rope"`로 받은 K가 HF의 `DynamicCache.layers[i].keys`와 정확히 일치하는지 + `kv_form="pre_rope"`로 받은 K에 RoPE를 적용한 결과가 같은 것과 일치하는지 검증.

- `docs/design-decisions.md` — Phase 1 결정 2건 추가 (KV capture form, decoder_layer 직접 호출).

### Tests

| Test | Result | Notes |
|---|---|---|
| `tests/test_layerwise.py::test_layerwise_matches_standard` | ✅ pass | **`max_diff = 0.000e+00`** — exact bit-identical (FP32 CPU, S=5, Qwen2.5-1.5B-Instruct) |
| `tests/test_layerwise.py::test_layerwise_matches_standard_longer` | ✅ pass | `max_diff < 1e-5` (assertion threshold; print suppressed in this run, but assertion held), S≈75 |
| `tests/test_layerwise.py::test_kv_extraction` | ✅ pass | post-RoPE K equals HF cache exactly; pre-RoPE K rotates to HF cache with `max_diff < 1e-6` per layer |
| `python scripts/verify_phase.py --phase 1` | ✅ pass | model.py 존재 + test_layerwise.py 통과 |

```
======================== 3 passed in 518.69s (0:08:38) =========================
```

### Acceptance criteria checklist

- [x] `LayerwiseModel`의 모든 메서드 구현 (`embed_tokens`, `prefill_layer`, `final_norm_and_lm_head`, `compute_position_embeddings`, properties)
- [x] `tests/test_layerwise.py::test_layerwise_matches_standard` 통과 (max_diff = 0.000e+00, FP32 CPU)
- [x] `tests/test_layerwise.py::test_kv_extraction` 통과 (pre-RoPE & post-RoPE 양쪽)
- [ ] (Optional) Mistral-7B 검증 — **미수행**, vast.ai에서 후속 진행 (아래 "Mistral-7B 후속 계획" 참조)
- [x] `python scripts/verify_phase.py --phase 1` 통과

## Decisions made

(전체는 `docs/design-decisions.md` 참조)

- **KV capture form: pre-RoPE primary via `k_proj` forward hook**. 후보 4개 중 hook 방식 선정 — 모델 동작에 영향을 주지 않는 observe-only이므로 bit-exact가 자동 보존되고, Llama/Mistral/Qwen2/Qwen2.5가 모두 `k_proj` 같은 이름의 submodule을 공유하므로 generic. Phase 2에서 chunk K 저장에 그대로 사용 가능.
- **HF의 `decoder_layer.forward` 그대로 호출 + `DynamicCache` 사용**. LMCache처럼 input_layernorm/qkv_proj/MLP를 일일이 풀어쓰지 않음 — per-arch 코드 중복을 피하고 bit-exact를 자동 보장. 비용: transformers ≥ 4.45의 layer signature 안정성에 의존 (현재 pin: `>=4.45,<5.0`).

## Deviations from plan

- **`test_layerwise_matches_standard_longer`을 `@pytest.mark.slow`로 마킹**. Mac CPU FP32에서 ~75 tokens × 28 layers × 1.5B params forward는 ~5분이 걸린다. 기본 `pytest tests/`에서는 빠른 sanity (5 tokens)만 돌리고, 긴 입력 테스트는 `pytest -m slow`로 옵트인. tasks/phase-1-layerwise-forward.md에는 명시 없었지만 실용적 분리.
- **사전 작업 1 — `.gitignore` 수정**. 원본 `.gitignore`의 `external/        # cloned LMCache repo etc.` 라인이 인라인 주석 때문에 패턴이 작동하지 않아 `external/LMCache`가 staged 됐음. 주석을 별도 줄로 분리하는 단순 fix를 같은 initial commit에 포함.
- **사전 작업 2 — `docs/prompts/` 인프라**. 별도의 두 번째 commit `Archive prompt: phase-1-layerwise-forward + prompt-archive infra`로 분리해 관리 (CLAUDE.md 규칙 그대로).

## Open questions / blockers

1. **Mistral-7B 검증**: Phase 1 task의 optional 항목. Mac 16GB RAM으로는 FP32 메모리 부담 (28GB)이 커 실행 불가. **후속 계획**: Phase 4에서 vast.ai 인스턴스를 띄울 때 같은 `tests/test_layerwise.py::test_layerwise_matches_standard`를 `parametrize`로 Mistral-7B에도 적용해 한 번에 검증. `LayerwiseModel`은 architecture-agnostic하게 작성됐으므로 model_id만 바꾸면 동작할 것으로 예상 — Mistral도 Llama/Qwen2와 동일한 decoder layer 시그니처 사용.
2. **Qwen3 등 다른 arch에서 hook 안전성**: Qwen3는 `k_proj` 후 `k_norm` 단계가 추가됨 (`docs/lmcache-analysis.md`의 LMCQwen3Model 참고). 현재 hook은 `k_proj.output`을 잡으므로 그건 "post-k_norm pre-RoPE"가 아닌 "pre-k_norm pre-RoPE"가 됨. Phase 2 RoPE 보정 시 k_norm을 한 번 적용해야 함. 현재 Qwen2.5에는 k_norm이 없어 문제없지만 design-decisions.md에 명시.

## transformers 4.57.6에서 까다로웠던 점

1. **`position_embeddings=(cos, sin)` 인자가 필수**. 4.40~4.45 시절에는 layer 내부에서 `rotary_emb`을 호출했지만, 4.57은 `Qwen2Model.forward`에서 한 번 계산해 모든 layer에 공유. layer signature는 default가 `None`이지만 `apply_rotary_pos_emb(query_states, key_states, cos, sin)` 직전에 unpack을 하므로 누락 시 silent하게 잘못된 logits가 나온다. 우리는 `compute_position_embeddings`로 standard와 동일하게 계산해 전달.
2. **`past_key_values` (복수형 's') vs `past_key_value` (단수)**. 4.57은 `@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")` — 단수형은 곧 사라진다. 4.58 호환 위해 우리는 처음부터 복수형만 사용.
3. **`create_causal_mask` 호출의 sliding-window 분기**. `Qwen2Model.forward`는 `causal_mask_mapping = {"full_attention": ..., "sliding_attention": ...}`을 만들어 layer의 `attention_type`에 맞춰 dispatch한다. Qwen2.5-1.5B는 모든 layer가 `full_attention`이라 분기가 필요 없어 단일 mask만 만든다. Qwen2 family에서 sliding 모델 (e.g., Qwen2-0.5B 일부)을 쓰게 되면 dispatch가 필요해진다.
4. **`DynamicCache(config=...)`의 새 시그니처**. 4.57은 config-aware. 빈 cache에 대해서도 `config`를 넘겨야 hybrid/sliding cache 구조를 정확히 초기화한다. 옛날 코드에서 `DynamicCache()` 만 호출하던 패턴은 안 됨.

## Numbers

| Setup | max logit diff vs `model(...).logits` |
|---|---|
| Qwen2.5-1.5B-Instruct, FP32, CPU, S=5 | **0.000e+00** (exact) |
| Qwen2.5-1.5B-Instruct, FP32, CPU, S≈75 | < 1e-5 (assertion threshold; assertion held) |
| Qwen2.5-1.5B-Instruct, FP32, CPU, KV per-layer | post-RoPE K: **exact match**; pre-RoPE K rotated to HF cache: < 1e-6 |

벽시계: `pytest tests/test_layerwise.py -v` 전체 (3 tests, slow 포함) ≈ 8분 38초 (Mac CPU, FP32). 그 중 모델 로드 ~16s, fast 2 tests ~3분, slow test ~5분.

## Files changed

```
CLAUDE.md                                 |  +9 (Prompt archiving 섹션 추가)
docs/prompts/README.md                    |  +53 (신규)
docs/prompts/phase-0-bootstrap.md         |  +32 (신규, 백필)
docs/prompts/phase-1-layerwise-forward.md | +110 (신규, 본 phase 프롬프트 원문)
docs/design-decisions.md                  |  +38 (Phase 1 결정 2건 추가)
src/cacheblend/model.py                   | +296 (LayerwiseModel 전체 구현)
tests/test_layerwise.py                   | +140 (3개 테스트)
.gitignore                                |  +1/-1 (external/ 인라인 주석 fix)
```

## Next phase prep

Phase 2를 시작할 때:
- `LayerwiseModel.prefill_layer` returns `LayerOutput(hidden, k, v)`. K의 form은 `kv_form` 인자로 결정 — Phase 2의 chunk store는 `kv_form="pre_rope"`로 받아 위치-무관 형태로 저장하는 것이 가장 단순하다 (RoPE 보정은 fuse 시점 한 번).
- `LayerwiseModel.compute_position_embeddings(hidden, position_ids)`로 임의의 position 범위에 대한 (cos, sin)를 즉시 얻을 수 있다 → chunk를 새 위치에 fuse할 때 그 위치의 position_embeddings를 호출해 K에 적용하면 끝.
- `attention_mask`는 `build_causal_mask`로 만든다. Phase 2의 cross-chunk attention은 chunk 경계에서 mask 패턴이 비표준이 될 수 있어, fusor 단계에서 "전체 시퀀스에 대한 standard causal mask"를 만들고 그걸 그대로 layer 호출에 넣으면 충분하다.
- `DynamicCache(config=...)`를 매 layer 직전에 fresh로 만들면 K/V를 layer마다 깨끗하게 추출할 수 있다. Phase 2는 cached chunk의 K/V를 cache에 prepopulate하는 시도가 필요할 수 있는데, 그건 fusor 측에서 직접 만든 KV로 attention을 호출하는 형태로 우회 가능 (cache update 우회).

## Mistral-7B 후속 계획

vast.ai 인스턴스 (RTX 4090 또는 동급) 한 시간이면 충분. 작업:
1. `bash scripts/vast.sh up <offer_id>` → push
2. `pytest tests/test_layerwise.py::test_layerwise_matches_standard -v -k mistral` (test parametrize 추가 후)
3. 결과 첨부해 Phase 1 report 보강 또는 Phase 2 report 서두에 한 줄 추가

이 작업은 Phase 4 도입부에서 다른 vast.ai 작업과 묶어 한 번에 진행하는 것이 비용/시간 측면에서 효율적.

## GitHub PR

PR URL: (커밋/푸시 후 본 보고서 발송 전에 채워짐. 발송 시점 메일 끝에 별도 메모로 첨부 또는 보고서 푸시 후 갱신)

## Suggested next prompt for Claude Code

> Phase 1 완료. main으로 PR을 머지한 뒤 Phase 2를 진행하세요.
>
> 다음 파일을 먼저 읽으세요: `tasks/phase-2-kv-storage.md`, `docs/paper-summary.md` (RoPE 섹션, §position recovery), `docs/design-decisions.md` (Phase 1 결정 2건 — pre-RoPE 저장은 이미 가능).
>
> Phase 1의 핵심 컨텍스트:
> - `LayerwiseModel.prefill_layer`는 pre-RoPE 또는 post-RoPE K를 모두 반환할 수 있음. Phase 2 chunk store는 `kv_form="pre_rope"`로 받아 저장. fuse 시점에 `compute_position_embeddings(hidden, target_pos)`로 새 위치의 (cos, sin)을 받아 RoPE 한 번 적용.
> - `tests/test_layerwise.py::test_layerwise_matches_standard_longer`는 `@pytest.mark.slow`. Phase 2의 단순 chunk concat 테스트는 5 토큰 chunk × N 정도로 작게 잡아 slow 마킹 없이 빠르게 통과되도록 할 것.
> - Phase 2의 acceptance: "동일 prefix 케이스에서 full reuse 결과가 full recompute와 logit 일치". 그 외 케이스에서는 cross-attention 차이로 logit이 다르며 OK.

---

Prompt archive: `docs/prompts/phase-1-layerwise-forward.md`
