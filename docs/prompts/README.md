# Prompt Archive

> 사용자(human)가 Claude Code에 보낸 작업-개시 프롬프트의 영구 기록.

## 디렉토리 목적

`reports/phase-N-report.md`가 "Claude가 뭘 했는가"의 기록이라면, 이 디렉토리는 "사용자가 뭘 시켰는가"의 기록입니다. 두 짝이 모이면 **의사결정 traceability**가 확보됩니다 — 나중에 "왜 Phase 3에서 X를 하지 않았지?" 같은 질문이 나오면, 그 시점 사용자의 원본 지시문으로 거슬러 올라갈 수 있습니다.

## 파일명 규칙

- 기본: `phase-N-<short-slug>.md`
  - 예: `phase-0-bootstrap.md`, `phase-1-layerwise-forward.md`, `phase-2-kv-storage.md`
- 한 phase 내 여러 프롬프트가 있으면 알파벳 suffix:
  - `phase-1a-layerwise-forward.md` (최초 지시)
  - `phase-1b-rope-shift-followup.md` (중간 추가 지시)
  - `phase-1c-bit-exact-tighten.md` (또 다른 추가 지시)
- slug는 짧고 식별 가능한 명사/동사구. 케밥-케이스, 영문 소문자.

## 파일 양식

상단에 YAML-style frontmatter, 그 아래에 **사용자 메시지 원문 그대로** (오타·강조·이모지 등 어떤 것도 편집하지 않음).

```markdown
---
date: YYYY-MM-DD
phase: <int>
topic: "한 줄 요약"
source: user via Claude chat
---

(사용자 프롬프트 원문 — 한 글자도 바꾸지 않음)
```

frontmatter는 정보 검색용 메타데이터이고, 본문은 1차 사료입니다. 본문은 절대 손대지 않습니다.

## 누가 언제 추가하는가

**Claude Code가 새 작업-개시 프롬프트를 받자마자 가장 먼저** 이 파일을 만듭니다.

> 작업-개시 프롬프트 = 코드를 변경하거나 phase를 진행시키는 지시. 단순 질문/clarification은 아카이브하지 않음.

순서:
1. `docs/prompts/phase-N-<slug>.md` 작성 (frontmatter + 원문).
2. **별도의 작은 commit**: `git add docs/prompts/... && git commit -m "Archive prompt: phase-N-<slug>"`.
3. 그 다음에 본 작업 시작.

이 순서를 지키는 이유: 본 작업 commit과 archive commit을 분리하면, 나중에 `git log -- docs/prompts/`만 보아도 "사용자가 어느 시점에 어떤 지시를 했는가" 타임라인이 깔끔하게 떨어집니다.

## 보관 정책

- **편집 금지**: 본문은 한 번 작성되면 수정하지 않습니다. 후속 보충 지시는 새 파일로 추가 (`-followup`, `-correction`).
- **삭제 금지**: 잘못된 파일명이라도 `git mv`로 이름만 변경하고 commit 메시지에 사유 기록.
- **요약본 별도**: 짧게 정리한 요약이 필요하면 `reports/phase-N-report.md`의 "Summary" 섹션에 작성. 원본 프롬프트는 그대로 둡니다.
