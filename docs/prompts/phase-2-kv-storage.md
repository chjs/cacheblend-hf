---
date: 2026-05-06
phase: 2
topic: "Merge PR #1 + Phase 2 KV storage"
source: user via Claude chat
---

PR #1 (https://github.com/chjs/cacheblend-hf/pull/1)을 main에 머지하고, 이어서 Phase 2를 진행하세요.

## 사전 — 프롬프트 아카이브

본 프롬프트를 `docs/prompts/phase-2-kv-storage.md`에 frontmatter(date, phase=2, topic="Merge PR #1 + Phase 2 KV storage", source="user via Claude chat")와 함께 저장. 별도 commit `Archive prompt: phase-2-kv-storage`로 push.

## Step 1 — PR #1 머지

1. `gh pr checks 1 --repo chjs/cacheblend-hf` 로 CI 상태 확인.
   - 모든 check 통과(녹색)면 다음 단계.
   - 실패면 stop & 사용자에게 보고 (어느 job/step에서 실패했는지 로그 일부 포함).
2. `gh pr merge 1 --repo chjs/cacheblend-hf --squash --delete-branch` 로 squash merge + 원격 브랜치 삭제.
3. 로컬 동기화: `git checkout main && git pull && git branch -d phase-1-layerwise-forward` (로컬 브랜치도 정리).

## Step 2 — Phase 2 시작

브랜치: `git checkout -b phase-2-kv-storage`

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
- `test_rope_shift_correctness`: RoPE shift된 K가 같은 위치에서 처음부터 prefill한 K와 일치.
- `test_full_reuse_matches_full_recompute_when_only_one_chunk_at_prefix`: 단일 prefix 청크 케이스에서 full reuse == full recompute.
- `test_full_reuse_diverges_with_multiple_chunks`: 다중 청크 케이스에서 logit이 측정 가능하게 다름.

### 제약
- `@torch.compile` 금지. 새 의존성 금지.
- Default 테스트는 5~20 토큰 chunk × 2~3개로. 모델 로딩 필요 테스트는 `@pytest.mark.requires_model`. 무거운 건 `@pytest.mark.slow`.
- 3번 시도 후 막히면 stop & 보고.

### Phase 1 long-test max_diff 측정 (한 번만)
`pytest tests/test_layerwise.py::test_layerwise_matches_standard_longer -v -s -m "slow and requires_model"` 실행 → 출력의 max_diff를 Phase 2 보고서의 "Phase 1 회귀" 섹션에 기록. 이 수치가 Phase 2~3의 base error budget 기준선.

### 마무리
- `pytest -v -m "requires_model and not slow"` 통과 + `pytest -v -m "not gpu and not slow and not requires_model"` 통과 + `python scripts/verify_phase.py --phase 2` 통과.
- `reports/phase-2-report.md` 작성. 보고서 마지막에 "Prompt archive: docs/prompts/phase-2-kv-storage.md" cross-ref. 명시 항목: RoPE shift 구현 방법, KVStore disk backend 여부, divergence 테스트의 logit L2 distance, Phase 1 long-test max_diff, GitHub PR URL.
- `python scripts/update_status.py --phase 2 --status completed`
- `git commit -m "Phase 2: KV storage & full reuse with RoPE recovery"` & `git push -u origin phase-2-kv-storage`
- `gh pr create --repo chjs/cacheblend-hf --base main --head phase-2-kv-storage --title "Phase 2: KV storage & full reuse" --body-file reports/phase-2-report.md` 로 PR 자동 생성.
- `python scripts/send_report.py --phase 2`

시작 전 30초 stop-and-think: GOAL.md 재확인. "Phase 2의 본질은 RoPE-aware full reuse이며 cross-attention 복원은 Phase 3의 일이다."
