# 🤖 Instructions for Claude Code

> 이 저장소는 **CacheBlend을 HuggingFace Transformers 위에 구현**하기 위한 작업 하네스입니다. 이 파일은 Claude Code가 작업할 때 따라야 할 운영 규칙을 정의합니다.

---

## ⚡ Every session starts here

1. **읽어야 하는 파일 (순서대로)**:
   - `GOAL.md` — 변경 금지된 최종 목표
   - `PHASES.md` — 전체 로드맵과 현재 phase
   - `reports/STATUS.md` — 직전 phase에서 무엇이 끝났는지
   - 사용자가 지시한 phase의 `tasks/phase-N-*.md`
2. 사용자가 명시적으로 phase를 지정하지 않으면, `reports/STATUS.md`에서 다음 phase를 추론하고 사용자에게 확인을 요청한다.
3. Phase 지시를 받으면, **그 phase의 task 파일에 정의된 acceptance criteria만** 만족시키는 작업을 수행한다.

## 🚧 Hard rules (위반 금지)

1. **목표를 벗어나지 않는다.** `GOAL.md`의 Non-goals에 해당하는 작업은 하지 않는다. 사용자가 그것을 명시적으로 요구하면, 먼저 GOAL.md와의 충돌을 알리고 확인을 받는다.
2. **단순함을 유지한다.** 새 의존성 추가는 정당화 필요. `requirements.txt`에 없는 것을 쓰려면 `docs/design-decisions.md`에 이유를 기록한다.
3. **테스트 없이는 phase를 끝내지 않는다.** 각 phase의 `tasks/phase-N-*.md`에 명시된 테스트가 모두 통과해야 한다.
4. **bit-exact 검증이 필요한 곳에서는 우회하지 않는다.** Phase 1의 layerwise forward는 표준 forward와 logit 차이 ≤ 1e-5 (FP16에서는 1e-3)이어야 한다. 통과 못하면 다음 phase로 넘어가지 않는다.
5. **LMCache 코드를 그대로 복사하지 않는다.** 참고는 하되, 우리는 더 단순한 버전을 새로 쓴다. 어떤 부분을 차용했는지 `docs/lmcache-analysis.md`에 명시한다.

## 🗂 Prompt archiving (every new user prompt)

When the user sends a new prompt that initiates work (anything beyond a clarifying question), Claude Code:
1. Before doing anything else, create `docs/prompts/phase-<N>-<slug>.md` with the full prompt text (no edits) and frontmatter (date, phase, topic).
2. Commit it as a separate small commit: `git add docs/prompts/... && git commit -m "Archive prompt: phase-N-<slug>"`.
3. Then start the actual work.

The archived prompt is the canonical record. Boil it down in your own summary if you want, but don't mutate the original.

## 📝 Phase 완료 절차

각 phase 작업이 끝났다고 판단되면:

```bash
# 1. 모든 테스트 실행
pytest tests/ -v

# 2. phase 검증 스크립트 실행 (해당 phase의 acceptance criteria 자동 점검)
python scripts/verify_phase.py --phase N

# 3. 보고서 작성: reports/phase-N-report.md
#    템플릿: reports/TEMPLATE.md 참고

# 4. STATUS 갱신
python scripts/update_status.py --phase N --status completed

# 5. 이메일 발송
python scripts/send_report.py --phase N
```

이 5단계가 끝난 뒤에만 사용자에게 "Phase N 완료" 라고 보고한다.

## 📊 Report 작성 가이드

`reports/phase-N-report.md`에는 반드시 다음을 포함한다:

- **What was done**: 구현한 모듈/함수 목록과 각각의 역할
- **Tests passed**: 어떤 테스트가 어떤 결과로 통과했는지 (수치 포함)
- **Decisions made**: 설계 갈림길에서 어떤 선택을 했는지, 왜 그랬는지
- **Deviations from plan**: 원래 task와 다르게 한 부분이 있다면 이유와 함께
- **Open questions**: 사용자 결정이 필요한 사항
- **Next phase prep**: 다음 phase를 시작할 때 알아야 할 컨텍스트
- **Files changed**: 핵심 변경 파일 목록 (git diff --stat)

이메일을 받는 사용자가 "다음 무슨 지시를 할지" 결정할 수 있을 만큼 자세하게 적는다.

## 🛑 막혔을 때

- **3번 시도 후에도 실패하면 멈춘다.** 보고서에 "Blocked: ..." 섹션을 적고 사용자에게 이메일을 보낸다.
- **GOAL.md와 충돌하는 요구**가 보이면 먼저 사용자에게 확인한다.
- **테스트가 자꾸 깨지면 단순화를 의심한다.** 더 작은 모델, 더 짧은 입력으로 좁혀서 디버그한다.

## 🧰 권장 도구 사용

- 모델 다운로드: `huggingface_hub`. 토큰은 `.env`의 `HF_TOKEN`.
- 빠른 반복용 모델: `Qwen/Qwen2.5-1.5B-Instruct` (논문 충실 재현은 `mistralai/Mistral-7B-Instruct-v0.2`)
- 테스트 데이터: 처음에는 합성 데이터 → Phase 5에서만 실제 벤치마크 데이터셋

## 🔁 Git 워크플로우

- Phase 단위로 브랜치를 만든다: `phase-N-<short-name>`
- 작은 commit, 의미 있는 메시지
- Phase 완료 시 main으로 PR. PR 본문에 보고서 요약 첨부
- 태그: phase 완료 후 `phase-N-complete` 태그 (GitHub Actions가 이메일 발송 트리거)
