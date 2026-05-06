---
date: 2026-05-06
phase: 1
topic: "Phase 1 instructions: git push, prompt archiving rule, pre-RoPE→post-RoPE fallback decision, layerwise forward implementation"
source: user via Claude chat
---

Phase 0 완료 보고 받았습니다. Phase 1로 진행하기 전 사전 작업 4건과 결정 정정 1건 후 본 작업을 시작하세요.

## 사전 작업 1 — Git 초기화 + GitHub 연결

빈 GitHub 저장소가 준비되어 있습니다: https://github.com/chjs/cacheblend-hf

다음을 순서대로 실행:
1. `git init`
2. `git checkout -b main`
3. `git add .` 후 `git status` 로 .env, external/LMCache, .venv, __pycache__, *.egg-info 가 staged 되지 않았는지 확인 (모두 .gitignore에 등록되어 있어야 함). staged 되어 있다면 .gitignore 점검 후 unstage.
4. `git commit -m "Initial harness + Phase 0 complete

- Harness scaffolding (GOAL/CLAUDE/PHASES/tasks/scripts/etc.)
- vast.ai integration addon (WORKFLOW.md, scripts/vast.sh, docs/VAST_GUIDE.md)
- Phase 0 deliverables (paper-summary.md 보강, lmcache-analysis.md)"`
5. `git remote add origin https://github.com/chjs/cacheblend-hf.git` (또는 SSH 인증 사용 중이면 `git@github.com:chjs/cacheblend-hf.git`)
6. `git push -u origin main`

푸시가 인증 문제로 실패하면 멈추고 보고. 어느 인증 방식이 잡혀 있는지(gh CLI, SSH key, PAT 등) 확인 후 사용자에게 안내.

푸시 성공 후:
7. `git checkout -b phase-1-layerwise-forward`

## 사전 작업 2 — 프롬프트 아카이빙 규칙 도입

앞으로 사용자가 Claude Code에 입력하는 프롬프트를 모두 `docs/prompts/` 에 보관합니다. 다음을 수행:

1. 디렉토리 생성: `mkdir -p docs/prompts`
2. `docs/prompts/README.md` 새로 작성. 다음 내용 포함:
   - 디렉토리 목적: 사용자 프롬프트의 영구 기록. Phase 보고서와 짝을 이뤄 의사결정 traceability 확보.
   - 파일명 규칙: `phase-N-<short-slug>.md` (예: `phase-0-bootstrap.md`, `phase-1-layerwise-forward.md`). 같은 phase에 여러 프롬프트가 있으면 `phase-1a-...`, `phase-1b-followup-...` 등 알파벳 suffix.
   - 파일 양식: 상단 YAML-style frontmatter (date, phase, topic, source: "user via Claude chat"), 그 아래에 프롬프트 원문 그대로 (어떤 편집도 하지 않음 — 오타 포함).
   - 누가 추가하는가: Claude Code가 새 프롬프트를 받을 때마다 작업 시작 전 가장 먼저 해당 파일을 만들고 commit. 프롬프트 본문은 사용자가 그 시점에 보낸 메시지 전체.
3. 회고적으로 Phase 0와 Phase 1 프롬프트 2개를 백필:
   - `docs/prompts/phase-0-bootstrap.md` 작성. 본문은 다음 한 줄로 대체하고 메모만 남김:
     ```
     "이 저장소는 CacheBlend을 HF transformers 위에 구현하는 프로젝트입니다. 먼저 GOAL.md, CLAUDE.md, PHASES.md, WORKFLOW.md, reports/STATUS.md를 읽고 Phase 0를 진행하세요." (재구성, 원문 정확히 보관 안 됨)
     ```
   - `docs/prompts/phase-1-layerwise-forward.md` 작성. 본문은 **현재 이 프롬프트(지금 사용자가 보낸 메시지) 전체**를 그대로 복사. frontmatter에 phase=1, topic="Phase 1 instructions with git push and prompt archiving rule" 등 기록.
4. `CLAUDE.md`의 "📝 Phase 완료 절차" 섹션 위에 다음 새 섹션을 삽입:

   ## 🗂 Prompt archiving (every new user prompt)

   When the user sends a new prompt that initiates work (anything beyond a clarifying question), Claude Code:
   1. Before doing anything else, create `docs/prompts/phase-<N>-<slug>.md` with the full prompt text (no edits) and frontmatter (date, phase, topic).
   2. Commit it as a separate small commit: `git add docs/prompts/... && git commit -m "Archive prompt: phase-N-<slug>"`.
   3. Then start the actual work.

   The archived prompt is the canonical record. Boil it down in your own summary if you want, but don't mutate the original.

## 사전 작업 3 — 환경 버전 메모

현재 transformers 4.57.6은 LlamaDecoderLayer / MistralDecoderLayer의 forward 시그니처가 4.40~4.45 시절과 다를 수 있습니다 (`position_embeddings=(cos, sin)` 사전 계산 인자, `cache_position` 인자). 작업 시작 전 해당 모듈 소스를 한 번 읽고, prefill_layer가 모든 필수 인자를 정확히 전달하도록 합니다. 인자 누락은 silent하게 잘못된 logits를 만듭니다.

## 사전 작업 4 — `tests/` 가 import cacheblend 가능한지 확인

Phase 0에서 editable install (`pip install -e .`)을 했으므로 `import cacheblend` 가 됩니다. `pytest tests/test_layerwise.py -v` 가 import error 없이 collect되는지 먼저 확인 후 본 작업 시작.

## 결정 정정 — Pre-RoPE 저장 방침

Phase 0 Decisions의 "K를 pre-RoPE로 저장" 결정을 다음과 같이 조정합니다:

- 1차 시도: 보고서대로 pre-RoPE 저장. HF attention 내부에서 `apply_rotary_pos_emb` 직전 K를 가로채는 hook 또는 wrap 구현.
- 시도 1회 안에 bit-exact 테스트(FP32 max_diff < 1e-5)가 통과하지 않으면 즉시 post-RoPE 저장으로 전환. Phase 2에서 위치 보정 시 inverse rotate + forward rotate 2회 적용. 정확성에는 영향 없음.
- 어떤 경로로 갔든 `docs/design-decisions.md`에 결정과 이유를 기록.

이 정정의 목적: Phase 1의 본질은 "표준 forward와 bit-exact한 layerwise 호출"이지 "pre-RoPE 저장"이 아닙니다. 후자에 매여 Phase 1을 지연시키지 마세요.

## 본 작업 — Phase 1

다음 파일을 순서대로 읽으세요:
1. `tasks/phase-1-layerwise-forward.md`
2. `docs/lmcache-analysis.md` (특히 LMCBaseModel.compute_layer 분석 — 우리는 vLLM 객체가 아닌 HF 표준 decoder_layer 호출로 다시 씀)
3. transformers 4.57.6의 `LlamaDecoderLayer.forward` / `MistralDecoderLayer.forward` 시그니처

목표:
- `src/cacheblend/model.py`의 `LayerwiseModel` 구현
- `tests/test_layerwise.py::test_layerwise_matches_standard` 통과 (Qwen2.5-1.5B-Instruct, FP32, CPU, max_diff < 1e-5)
- `test_kv_extraction` 통과 — 추출한 KV가 (위 결정에 따라) pre-RoPE 또는 post-RoPE 형태로 일관되게 저장 가능함을 검증
- Mistral-7B 검증은 vast.ai를 띄워야 하므로 이번 phase에서는 skip 가능 (단, 코드는 model architecture에 종속되지 않게 일반화)

제약:
- `@torch.compile` 사용 금지
- 새 의존성 추가 금지 (정당화 시 design-decisions.md 기록)
- 3번 시도 후 막히면 멈추고 보고

완료 시:
- `pytest tests/test_layerwise.py -v` 전체 통과 확인
- `python scripts/verify_phase.py --phase 1` 통과 확인
- `reports/phase-1-report.md` 작성 (TEMPLATE.md 양식). 보고서 마지막에 "Prompt archive: docs/prompts/phase-1-layerwise-forward.md" 한 줄 cross-reference 추가.
- `python scripts/update_status.py --phase 1 --status completed`
- 변경사항 commit & push:
  * `git add -A`
  * `git commit -m "Phase 1: layerwise forward (max_diff=<수치>)"`
  * `git push -u origin phase-1-layerwise-forward`
- `python scripts/send_report.py --phase 1`
- 보고서에 다음을 명시:
  * 최종 채택한 K 저장 형태 (pre-RoPE vs post-RoPE)와 그 결정 경로
  * transformers 4.57.6 API에서 까다로웠던 부분
  * Mistral-7B 검증 미수행 사실과 vast.ai에서 후속 검증할 계획
  * GitHub PR URL (사용자가 main으로 머지)

시작하기 전 30초 stop-and-think: GOAL.md의 Non-goals를 다시 한 번 보고, "표준 forward와 bit-exact 일치" 목표 외 다른 데로 새지 않을 것을 확인.
