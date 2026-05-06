# Quick Start (for the human user)

> 이 문서는 **사용자(repo owner)** 가 처음 한 번만 읽는 셋업 가이드입니다.
> Claude Code는 `CLAUDE.md` 를 봅니다.

## 0. 사전 요구사항

- Python 3.10 이상
- Git
- (옵션) CUDA GPU + 적절한 CUDA toolkit
- Gmail 계정 + 2FA 활성화 + **앱 비밀번호** 1개

## 1. GitHub 저장소 만들기

1. github.com에서 새 저장소 생성: 이름 `cacheblend-hf` (private 권장)
2. 이 폴더에서:
   ```bash
   cd cacheblend-hf
   git init
   git add .
   git commit -m "Initial harness"
   git branch -M main
   git remote add origin git@github.com:<YOUR_USERNAME>/cacheblend-hf.git
   git push -u origin main
   ```

## 2. Gmail 앱 비밀번호 발급

1. Google 계정 → 보안 → 2단계 인증을 켭니다 (이미 켜져 있으면 통과)
2. https://myaccount.google.com/apppasswords 로 이동
3. "앱 이름"에 `cacheblend-hf` 같은 이름 입력 → 생성
4. 16자리 비밀번호가 나옴. **이걸 어딘가 복사해 두세요. 다시 못 봅니다.**

## 3. GitHub Secrets 설정 (이메일 자동 발송용)

GitHub 저장소 → Settings → Secrets and variables → Actions → "New repository secret":

- `GMAIL_ADDRESS`: 본인 Gmail 주소
- `GMAIL_APP_PASSWORD`: 위에서 받은 앱 비밀번호 (공백 없이 16자리)
- `REPORT_EMAIL_TO`: `ch.jungsik@gmail.com`

이렇게 하면 `phase-N-complete` 태그를 push 했을 때 자동으로 이메일이 갑니다.

## 4. 로컬 작업 환경

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

cp .env.example .env
# .env 파일을 열어 GMAIL_ADDRESS, GMAIL_APP_PASSWORD 등 채우기
# (HF_TOKEN은 Mistral-7B 같은 게이트된 모델 쓸 때만 필요)

pip install -r requirements.txt

# 이메일 파이프라인 dry-run 테스트
python scripts/send_report.py --phase 0 --dry-run
```

## 5. Claude Code에 작업 지시하기

Claude Code 세션에서 저장소를 열고, 가장 먼저:

> **첫 프롬프트 예시:**
>
> "이 저장소는 CacheBlend을 HF transformers 위에 구현하는 프로젝트입니다.
> 우선 GOAL.md, PHASES.md, CLAUDE.md를 읽고 현재 상태를 파악해 주세요.
> 그런 다음 Phase 0를 시작하세요. tasks/phase-0-analysis.md를 따르고,
> 끝나면 보고서를 작성해 reports/ 에 저장한 뒤
> `python scripts/send_report.py --phase 0` 을 실행해 주세요."

이후 phase 마다:

1. Claude Code가 작업 → 보고서 작성 → 이메일 발송
2. 당신은 받은 이메일을 Claude(채팅)에 붙여넣기
3. Claude가 다음 phase 프롬프트를 작성
4. 그 프롬프트를 Claude Code에 전달
5. 반복

## 6. 일반적인 트러블슈팅

| 증상 | 해결 |
|---|---|
| `SMTPAuthenticationError` | 앱 비밀번호 다시 확인. 정규 비밀번호는 안 됩니다. |
| `Cannot import torch` | `pip install -r requirements.txt` 다시 |
| Mistral-7B 다운로드 실패 | HF에서 모델 access 허락 필요. `HF_TOKEN` 발급 후 `.env` 에 |
| 테스트 너무 느림 | 작은 모델(`Qwen2.5-1.5B`) 부터 시작. CLAUDE.md 권장. |
| GitHub Action 이메일 안 옴 | Secrets 3개 모두 설정됐는지, 태그 형식이 `phase-N-complete` 인지 확인 |

## 7. Phase 진행 명령어 요약

```bash
# 작업 중
git checkout -b phase-N-...

# 작업 완료 시 (Claude Code가 자동으로 함)
pytest tests/ -v
python scripts/verify_phase.py --phase N
python scripts/update_status.py --phase N --status completed
python scripts/send_report.py --phase N

# 사용자가 main에 머지 후
git tag phase-N-complete
git push origin phase-N-complete   # GitHub Action 트리거
```
