---
date: 2026-05-06
phase: 2
topic: "CI fix + Phase 2 KV storage"
source: user via Claude chat
---

GitHub Actions CI가 Phase 1 push에서 ImportError로 실패 (run 25413053881). 원인: CI workflow에 `pip install -e .` 누락. 이번에 CI fix(Part A)와 Phase 2 본 작업(Part B)을 순서대로 진행하세요.

## 사전 — 프롬프트 아카이브 (CLAUDE.md 규칙)

본 프롬프트 전체를 그대로 `docs/prompts/phase-2-kv-storage-with-ci-fix.md`에 frontmatter(date, phase=2, topic="CI fix + Phase 2 KV storage", source="user via Claude chat")와 함께 저장. 별도 commit `Archive prompt: phase-2-kv-storage-with-ci-fix`로 push.

## 사전 점검 — Git 상태

`git branch --show-current` + `gh pr list --repo chjs/cacheblend-hf --state open` 로 PR #1 상태 확인.
- Case 1 (PR #1 머지됨, main에 phase-1 포함): `git checkout main && git pull && git checkout -b fix/ci-after-phase-1`에서 Part A. Part A push & PR 생성 후 곧바로 Part B로.
- Case 2 (PR #1 open): phase-1 브랜치에 Part A를 추가 commit으로 얹어 PR #1 보강. Part B는 stop, 사용자에게 머지 요청 후 종료.

## Part A — CI fix

1. `.github/workflows/ci.yml`의 install 스텝에 `pip install -e .` 한 줄 추가 (`pip install -r requirements.txt` 다음).
2. `pyproject.toml`의 `[tool.pytest.ini_options].markers`에 `"requires_model: tests that download or load HF models (skipped in CI)"` 추가.
3. `tests/test_layerwise.py`에서 `from_pretrained`를 호출하는 모든 테스트에 `@pytest.mark.requires_model` 추가.
4. `.github/workflows/ci.yml`의 pytest 명령을 `pytest -v -m "not gpu and not slow and not requires_model"`로 변경.
5. `tests/test_smoke.py` 신규 작성 — `import cacheblend` / `from cacheblend.model import LayerwiseModel` / `import torch` 3개 테스트 (모델 로딩 없이 1초 이내).
6. `CLAUDE.md`에 test markers (slow/gpu/requires_model) 설명 한 단락 추가.
7. `docs/design-decisions.md`에 "CI는 model-loading test를 skip" 결정 한 줄 추가.

검증:
- `pytest -v -m "not gpu and not slow and not requires_model"` → smoke 3개만 통과 (1초 이내).
- `pytest -v -m "requires_model and not slow"` → Phase 1 fast model test 회귀 없이 통과.

마무리: `git commit -m "CI: pip install -e + skip model-loading tests, add smoke tests"` 후 위 Case에 따라 push & PR.

## Part B — Phase 2: KV Storage & Full Reuse (Case 1만 진행)

브랜치: `git checkout main && git checkout -b phase-2-kv-storage`

읽어야 할 파일: `tasks/phase-2-kv-storage.md`, `docs/paper-summary.md`(RoPE & position recovery), `docs/design-decisions.md`, `ARCHITECTURE.md`.

### Phase 1 상속 컨텍스트
- `LayerwiseModel.prefill_layer`는 `kv_form="pre_rope" | "post_rope"` 둘 다 가능. Chunk store는 **`pre_rope`** 로 저장 (RoPE 보정은 fuse 시점 한 번).
- `LayerwiseModel.compute_position_embeddings(hidden, position_ids)`로 임의 위치의 (cos, sin) 즉시 획득 → fuse 시 K에 적용.
- `LayerwiseModel.build_causal_mask`로 전체 시퀀스 standard causal mask 만들어 layer에 전달. cross-chunk mask 별도 X.

### 구현 (tasks/phase-2-kv-storage.md의 acceptance와 동일)
- `src/cacheblend/chunker.py` — `Chunk` dataclass, `chunk_input()`. 해시는 텍스트만으로 (위치 무관).
- `src/cacheblend/kv_store.py` — `KVStore` (in-memory + 옵션 disk, K는 pre-RoPE).
- `src/cacheblend/rope.py` — `apply_rope_shift(...)`. model의 rotary_emb 재사용해 convention 자동 일치.
- `src/cacheblend/precompute.py` — `precompute_chunk_kv(model, chunk_text, tokenizer)`.
- `src/cacheblend/fusor.py::fuse_full_reuse(...)` — RoPE 보정 후 concat → 최종 logits.

### 테스트 (`tests/test_kv_reuse.py`)
- `test_rope_shift_correctness`: RoPE shift된 K가 같은 위치에서 처음부터 prefill한 K와 일치 (FP tolerance).
- `test_full_reuse_matches_full_recompute_when_only_one_chunk_at_prefix`: 단일 prefix 청크 케이스에서 full reuse == full recompute.
- `test_full_reuse_diverges_with_multiple_chunks`: 다중 청크 케이스에서 logit이 측정 가능하게 다름 (논문의 기대된 발산).

### 제약
- `@torch.compile` 금지. 새 의존성 금지.
- Default 테스트는 5~20 토큰 chunk × 2~3개로 빠르게. 모델 로딩 필요한 테스트는 `@pytest.mark.requires_model`. 무거운 건 `@pytest.mark.slow`.
- 3번 시도 후 막히면 stop & 보고.

### Phase 1 long-test 수치 채우기
Phase 1 보고서에 누락된 `test_layerwise_matches_standard_longer`의 정확한 max_diff를 한 번만 측정:
`pytest tests/test_layerwise.py::test_layerwise_matches_standard_longer -v -s -m "slow and requires_model"` 실행 → 출력의 max_diff를 Phase 2 보고서의 "Phase 1 회귀" 섹션에 기록. 이 수치가 Phase 2~3의 base error budget 기준선.

### 마무리
- `pytest -v -m "requires_model and not slow"` 통과 + `pytest -v -m "not gpu and not slow and not requires_model"` 통과 + `python scripts/verify_phase.py --phase 2` 통과.
- `reports/phase-2-report.md` 작성 (TEMPLATE.md 양식). 보고서 마지막에 "Prompt archive: docs/prompts/phase-2-kv-storage-with-ci-fix.md" cross-ref. 명시 항목: RoPE shift 구현 방법(rotary_emb 재사용 여부), KVStore disk backend 여부(미구현이면 Phase 4로 미룬다고 기록), divergence 테스트의 logit L2 distance 수치, Phase 1 long-test max_diff, GitHub PR URL.
- `python scripts/update_status.py --phase 2 --status completed`
- `git commit -m "Phase 2: KV storage & full reuse with RoPE recovery"` & `git push -u origin phase-2-kv-storage`
- `python scripts/send_report.py --phase 2`

시작 전 30초 stop-and-think: GOAL.md 재확인. "Phase 2의 본질은 RoPE-aware full reuse이며 cross-attention 복원은 Phase 3의 일이다."
