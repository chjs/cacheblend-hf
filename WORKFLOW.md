# Workflow — Day to Day

> 사용자(you)와 Claude Code가 어떤 흐름으로 일을 진행하는지. `QUICKSTART.md`는 1회성 셋업이고, 이 파일은 **반복적인 작업 흐름**을 다룹니다.

---

## 환경 한 장 요약

```
[갤럭시탭 / Claude 채팅]   ←→   [맥북 (Claude Code 실행)]   ←→   [vast.ai GPU 인스턴스]
       (지시 + 보고서 분석)              (Phase 0~3 작업)             (Phase 4~5 GPU 작업)
```

---

## 첫 1회 셋업 (맥북)

```bash
cd ~/projects/cacheblend-hf

python3 -m venv .venv
source .venv/bin/activate

# Mac에서는 CPU torch + MPS torch 둘 다 가능. 우선 표준 설치
pip install -r requirements.txt

cp .env.example .env
# .env 편집: GMAIL_ADDRESS, GMAIL_APP_PASSWORD, HF_TOKEN(있으면)

# 이메일 검증
python scripts/send_report.py --phase 0 --dry-run
# ✅ 메시지가 보이면 OK

# vast.ai 검증 (CLI 이미 인증돼 있다고 하셨으니)
vastai show instances   # 비어 있어도 OK
```

---

## 각 Phase 마다의 흐름

### A. 맥북에서 끝낼 수 있는 phase (0, 1, 2, 3)

1. **갤럭시탭에서 SSH로 맥북 접속** → Claude Code 실행
2. Claude Code에 지시:
   > "Phase N 진행해. 먼저 GOAL.md, PHASES.md, reports/STATUS.md, tasks/phase-N-*.md 읽고 작업해. 끝나면 reports/phase-N-report.md 작성하고 send_report.py로 발송해."
3. Claude Code가 작업 → 테스트 → 보고서 → 이메일
4. 갤럭시탭에서 **이메일 확인** (이게 핵심 — 화면 작아도 됨)
5. 갤럭시탭의 Claude 채팅에 그 보고서 붙여넣기 + "다음 Phase 프롬프트 만들어 줘" 요청
6. 받은 프롬프트를 다시 맥북의 Claude Code에 입력 → 다음 Phase

### B. vast.ai가 필요한 phase (4, 5)

1. 맥북에서 인스턴스 띄우기:
   ```bash
   bash scripts/vast.sh search        # 후보 보기
   bash scripts/vast.sh up <OFFER_ID> # 선택해서 띄우기 (.vast_id 자동 저장)
   sleep 60
   bash scripts/vast.sh ssh           # 잘 떴는지 확인
   exit
   ```

2. 코드 동기화:
   ```bash
   bash scripts/vast.sh push          # 맥북 → vast
   ```

3. Claude Code에 vast 작업 지시:

   **Option A (Claude Code가 직접 ssh 실행)**: Claude Code에 `vast.sh ssh` 와 함께 원격 명령을 실행하도록 지시. 단, 한 번에 ssh로 들어갔다 나오는 형태가 좋음 (대화형 세션은 불안정).

   **Option B (사용자가 ssh 따로)**: Claude Code는 맥북에서 코드만 작성/수정. 사용자가 vast 인스턴스에 ssh로 접속해 `tmux` 로 벤치마크 실행. 결과를 `vast.sh pull` 로 회수.

   초보 단계에서는 **Option B 권장**. Claude Code가 작성한 벤치마크 스크립트를 직접 실행하는 게 가장 안전.

4. 작업 끝났으면:
   ```bash
   bash scripts/vast.sh pull     # 결과 회수
   bash scripts/vast.sh stop     # GPU 비용 멈춤, 디스크는 유지
   ```

5. 한참 안 쓸 거면 `destroy`. 며칠 내 다시 쓸 거면 `stop` 으로 두고 모델 캐시 보존.

---

## Phase 별 vast 사용 가이드라인

| Phase | 맥북에서 | vast.ai 필요? | 권장 인스턴스 |
|---|---|---|---|
| 0 | ✅ 전부 | ❌ | — |
| 1 | ✅ Qwen2.5-1.5B FP32 CPU 검증 | ⚠️ optional sanity check | RTX 3090 1시간이면 충분 |
| 2 | ✅ | ❌ | — |
| 3 | ✅ 작은 합성 입력 | ⚠️ Mistral-7B로 한 번 검증 권장 | RTX 4090 |
| 4 | 일부 | ✅ TTFT 측정 | RTX 4090 + NVMe |
| 5 | 분석/플롯만 | ✅ 본 벤치마크 | RTX 6000 Ada (48GB) 또는 A40 |

---

## 트러블슈팅 자주 하는 것

| 증상 | 원인 / 해결 |
|---|---|
| `vastai create` 후 ssh가 안 됨 | 30~60초 기다렸는지? `vastai logs <ID>` 로 onstart 출력 확인 |
| `HF_HOME` 적용 안 됨 (모델이 ~/.cache/huggingface 에 받힘) | onstart-cmd의 env >> /etc/environment가 새 SSH 세션에 반영됨. 같은 세션에서는 `export HF_HOME=/workspace/hf_cache` 직접 |
| rsync가 venv까지 보냄 | `vast.sh push` 가 `--exclude .venv` 함. 그래도 보내지면 .venv 위치 확인 |
| 갑자기 비용이 커짐 | 인스턴스가 destroy 안 되고 stop만 된 상태로 디스크 비용이 누적된 것일 수 있음. `vastai show instances` 로 확인 |
| Claude Code가 GOAL을 자꾸 잊음 | 매 세션 시작에 "GOAL.md 먼저 읽어" 명시. CLAUDE.md 첫 단락이 정확히 그 지침 |

---

## 다음 액션 (지금 해야 할 일)

1. 위 "첫 1회 셋업" 실행
2. Claude Code 켜고 다음 프롬프트로 시작:

   ```
   이 저장소는 CacheBlend을 HF transformers 위에 구현하는 프로젝트입니다.

   먼저 다음 파일들을 순서대로 읽으세요:
   1. GOAL.md
   2. CLAUDE.md
   3. PHASES.md
   4. WORKFLOW.md
   5. reports/STATUS.md

   읽고 나면 현재 상태와 Phase 0의 작업 범위를 한 단락으로 요약해 주세요.
   요약을 본 뒤 제가 본격 작업을 지시하겠습니다.
   ```

3. 요약이 적절하면 다음:

   ```
   좋습니다. tasks/phase-0-analysis.md에 따라 Phase 0를 진행하세요.
   끝나면 reports/phase-0-report.md를 작성하고
   `python scripts/send_report.py --phase 0` 으로 보고서를 발송해 주세요.
   ```

4. 갤럭시탭에서 이메일 도착 확인 → 본 채팅에 붙여넣기 → Phase 1 프롬프트 받기.
