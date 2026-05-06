---
date: 2026-05-06
phase: 3
topic: "Merge PR #2 + Phase 3 selective KV recompute"
source: user via Claude chat
---

Phase 2 PR #2 (https://github.com/chjs/cacheblend-hf/pull/2)을 머지하고 Phase 3 (Selective KV Recompute)를 진행하세요. 이번이 CacheBlend 논문의 핵심 구현입니다.

## 사전 — 프롬프트 아카이브

본 프롬프트를 `docs/prompts/phase-3-selective-recompute.md`에 frontmatter(date, phase=3, topic="Merge PR #2 + Phase 3 selective KV recompute") + 본문 그대로 저장. 별도 commit `Archive prompt: phase-3-selective-recompute`로 push.

## Step 1 — PR #2 머지

1. `gh pr checks 2 --repo chjs/cacheblend-hf` 로 CI 확인. 실패면 stop & 보고.
2. `gh pr merge 2 --repo chjs/cacheblend-hf --squash --delete-branch`
3. `git checkout main && git pull && git branch -d phase-2-kv-storage`

## Step 2 — Phase 3 시작

브랜치: `git checkout -b phase-3-selective-recompute`

읽어야 할 파일: `tasks/phase-3-selective-recompute.md`, `docs/paper-summary.md`(Insight 1/2, gradual filtering, pseudo-code), `docs/design-decisions.md`(Phase 2의 hook 기반 K/V swap 메커니즘), `external/LMCache/lmcache/v1/compute/blend/blender.py`(LMCache의 process_qkv 88-113줄 — squared L2 + topk 패턴 참고).

### Phase 2 상속 컨텍스트
- `fuse_full_reuse`의 layer 루프 + `k_proj`/`v_proj` forward hook이 selective recompute의 뼈대. 차이는 hook이 cached K/V 전체를 주입하는 대신 **HKVD 토큰만 fresh K/V로 두고 나머지는 cached**로 합성하여 출력.
- Phase 2 divergence baseline: 2 청크 S=20에서 L2 = 1239. Phase 3 목표: ratio=15%에서 L2 < 100 (elbow 진입), 가능하면 < 10.
- Phase 1 base error = 0 (FP32 CPU). Phase 3의 logit 차이는 100% selective recompute가 만든 차이로 해석.

### 구현 (`tasks/phase-3-selective-recompute.md` acceptance와 동일)

`src/cacheblend/hkvd.py` 신규:
- `kv_deviation(k_fresh, k_cached) -> Tensor[num_tokens]` — squared L2를 head/dim에서 reduce, 토큰별 스칼라.
- `select_top_k(deviation, k) -> Tensor[indices]` — torch.topk → torch.sort.
- `gradual_ratio_schedule(num_layers, target_ratio=0.15)` — Phase 3에서는 단일 ratio 단일 check layer로 시작. 함수는 정의하되 default는 모든 layer에 같은 ratio. Phase 5 F1 부족 시 decay 도입.

`src/cacheblend/fusor.py::fuse_selective(model, chunks, kv_store, recompute_ratio=0.15, check_layer=1)`:
- Layer 0..check_layer-1: full reuse (cached K/V 그대로 hook).
- Layer check_layer: fresh K_check 계산 → cached K_check와 deviation → top r% 인덱스 = HKVD set S.
- Layer check_layer+1..L-1: hook이 합성 K/V 출력 = cached K/V with positions in S replaced by fresh K/V (fresh는 그 layer의 partial prefill에서). LMCache처럼 S는 layer 간 동일하게 고정 (gradual narrowing 미적용 v1).
- 모든 cached chunk 토큰 외 (system/query 같은 uncached) 토큰은 매 layer에서 fresh.

### 테스트 (`tests/test_selective.py`)

- `test_recompute_ratio_zero_equals_full_reuse`: ratio=0 → fuse_full_reuse와 logit 동일 (max_diff < 1e-5).
- `test_recompute_ratio_one_equals_full_recompute`: ratio=1.0 → full recompute와 logit 동일 (max_diff < 1e-5).
- `test_selective_better_than_full_reuse`: 다중 청크 입력에서 ratio=0.15 selective의 L2(vs full recompute) ≤ full reuse L2 × 0.5. Phase 2의 L2=1239 시나리오 그대로 사용.
- `test_quality_vs_ratio` (parametrize 0.05/0.10/0.15/0.20): ratio 증가 시 L2 평균적으로 감소 (strict monotonic 요구 X).

모든 테스트 `@pytest.mark.requires_model`. 짧은 입력 유지 (S ≤ 30).

### 제약
- `@torch.compile` 금지. 새 의존성 금지.
- check_layer는 1로 고정 (LMCache default). 튜닝 X.
- gradual narrowing(layer마다 S 좁히기)은 v1에서 미구현. Phase 5에서 F1 부족 시 도입.
- 3번 시도 후 막히면 stop & 보고.

### 마무리
- `pytest -v -m "requires_model and not slow"` (Phase 1+2+3 모두) + `pytest -v -m "not gpu and not slow and not requires_model"` + `python scripts/verify_phase.py --phase 3` 통과.
- `reports/phase-3-report.md` 작성. 명시 항목:
  * ratio별 L2 표 (0%, 5%, 10%, 15%, 20%, 50%, 100%) — Phase 2 동일 입력 (S=20 두 청크) 기준
  * elbow 위치 (논문 Fig 6 재현 시도)
  * Insight 2 검증: layer 1 HKVD top-k와 layer L-1에서 측정한 deviation top-k의 Spearman rank correlation 또는 단순 overlap 비율
  * "ratio=0이 full reuse와 일치 / ratio=1이 full recompute와 일치" sanity 결과
  * GitHub PR URL
  * "Prompt archive: docs/prompts/phase-3-selective-recompute.md" cross-ref
- `python scripts/update_status.py --phase 3 --status completed`
- `git commit -m "Phase 3: selective KV recompute (HKVD selection at single check layer)"` & `git push -u origin phase-3-selective-recompute`
- `gh pr create --repo chjs/cacheblend-hf --base main --head phase-3-selective-recompute --title "Phase 3: selective KV recompute" --body-file reports/phase-3-report.md`
- `python scripts/send_report.py --phase 3`

시작 전 30초 stop-and-think: GOAL.md 재확인. "Phase 3의 본질은 cross-attention 복원이며, 단순한 단일 check layer + 단일 ratio로 Phase 2 baseline의 L2를 의미 있게 줄이는 것이 1차 목표. gradual narrowing/per-layer decay는 미래 작업."
