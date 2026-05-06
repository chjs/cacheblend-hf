# Phase 0 — Setup & Analysis

## Objective

작업 환경을 갖추고, 논문/LMCache 코드 분석 산출물을 만든다. 코드는 거의 작성하지 않는다.

## Inputs to read first

1. `GOAL.md`
2. `PHASES.md`
3. `docs/paper-summary.md` — 논문 핵심 정리 (이미 채워져 있음, 보강만 필요)
4. CacheBlend 논문 PDF (사용자가 따로 제공하면 그것을 사용. 없으면 arXiv에서 받음)

## Steps

### 0.1 Repo bootstrap

```bash
# Verify Python version
python --version  # >= 3.10

# Install
pip install -r requirements.txt

# Sanity import
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

성공 기준: import 에러 없이 버전 출력.

### 0.2 Email pipeline dry-run

```bash
cp .env.example .env
# Edit .env (사용자가 채워줘야 하는 항목들 — 채워지지 않았을 가능성도 있음)

python scripts/send_report.py --phase 0 --dry-run
```

`--dry-run` 은 SMTP 연결 시도까지만 하고 실제로는 보내지 않는다. 사용자가 `.env` 를 안 채웠으면 친절한 에러 메시지를 출력해야 한다.

### 0.3 Paper summary 보강

`docs/paper-summary.md` 를 처음부터 끝까지 읽고, 빠진 부분이 있으면 보강한다. 특히:
- Figure 6 (recompute ratio vs forward attention deviation) 의 의미
- Figure 8 (HKVD rank correlation between layers) 의 의미
- §5의 LoadingController 알고리즘 (수식 포함)

이 파일은 우리의 "single source of truth"이므로 정확해야 한다.

### 0.4 LMCache 분석

```bash
mkdir -p external
cd external
git clone --depth 1 https://github.com/LMCache/LMCache.git
cd ..
```

`docs/lmcache-analysis.md` 의 모든 섹션을 채운다. **코드를 줄 단위로 읽지 말고**, 모듈 단위로 빠르게 훑은 뒤 우리 아키텍처 (`ARCHITECTURE.md`) 와 어떻게 매핑되는지 위주로 정리한다.

특히 명확히 답해야 하는 것들:
1. LMCache의 HKVD 선택 함수 위치 (파일/함수명)
2. layer-by-layer 부분 prefill을 어디서 어떻게 하는가 (vLLM hook? custom forward?)
3. RoPE shift는 어디에 구현되어 있는가
4. KV 저장 단위 — 토큰? 청크? page?
5. recompute_ratio scheduling 코드 — 어떻게 layer마다 ratio를 정하는가

> 이 답들은 Phase 1~3 작업의 직접적 입력이 된다.

### 0.5 Initial repo structure 점검

다음 파일/폴더가 모두 존재하고 빈 skeleton이라도 있는지 확인:

- [ ] `src/cacheblend/__init__.py`
- [ ] `src/cacheblend/model.py`
- [ ] `src/cacheblend/kv_store.py`
- [ ] `src/cacheblend/fusor.py`
- [ ] `src/cacheblend/controller.py`
- [ ] `src/cacheblend/hkvd.py`
- [ ] `src/cacheblend/rope.py`
- [ ] `src/cacheblend/utils.py`
- [ ] `tests/test_layerwise.py`
- [ ] `tests/test_kv_reuse.py`
- [ ] `tests/test_selective.py`
- [ ] `tests/test_pipeline.py`
- [ ] `tests/test_e2e.py`

없으면 만든다 (최소 docstring + `pass`).

## Acceptance criteria

- [ ] `python -c "import torch, transformers"` 성공
- [ ] `python scripts/send_report.py --phase 0 --dry-run` 성공
- [ ] `docs/paper-summary.md` 에 빈 섹션 없음
- [ ] `docs/lmcache-analysis.md` 의 5가지 핵심 질문에 모두 답해짐
- [ ] 위 체크리스트의 모든 파일 존재

## Report (reports/phase-0-report.md)

`reports/TEMPLATE.md` 형식을 따라 작성. 특히:
- LMCache 분석에서 얻은 5가지 답
- LMCache 코드의 어떤 부분이 우리 작업에 가장 위협적인 복잡성인지 (예: vLLM 결합도)
- Phase 1을 시작하기 전에 사용자에게 확인받아야 할 사항이 있다면 명시
