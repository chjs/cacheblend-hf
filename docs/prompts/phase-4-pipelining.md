---
date: 2026-05-06
phase: 4
topic: "Merge PR #3 + long-chunk sanity + Phase 4 pipelining"
source: user via Claude chat
---

Phase 3 PR #3 (https://github.com/chjs/cacheblend-hf/pull/3)을 머지한 뒤, 사전 검증 1건과 Phase 4 (Pipelining) 본 작업을 진행하세요.

## 사전 — 프롬프트 아카이브

본 프롬프트를 `docs/prompts/phase-4-pipelining.md`에 frontmatter(date, phase=4, topic="Merge PR #3 + long-chunk sanity + Phase 4 pipelining") + 본문 그대로 저장. 별도 commit `Archive prompt: phase-4-pipelining`로 push.

## Step 1 — PR #3 머지

1. `gh pr checks 3 --repo chjs/cacheblend-hf` 로 CI 확인. 실패면 stop & 보고.
2. `gh pr merge 3 --repo chjs/cacheblend-hf --squash --delete-branch`
3. `git checkout main && git pull && git branch -d phase-3-selective-recompute`

## Step 2 — Long-chunk sanity (Phase 4 진입 전 보험)

브랜치: `git checkout -b phase-4-pipelining`

Phase 3에서 ratio=0.15 시 L2 감소 23%로 측정됨. 보고서는 "20-token 입력의 한계"로 해석. 이 해석이 옳은지 확인하는 한 번의 측정.

`benchmarks/phase3_sweep.py`를 참고해 `benchmarks/long_chunk_sanity.py`를 신규 작성:
- 동일 시나리오 (2 청크), 단 chunk B 길이를 50, 100, 200 토큰으로 확장.
- chunk A는 짧게 유지 (예: 10토큰).
- 각 길이에서 ratio ∈ {0.0, 0.05, 0.10, 0.15, 0.20, 0.50}의 L2 측정.
- 출력: chunk B 길이별 ratio-L2 표 + ratio=0.15에서의 reduction 비율.

기대 결과: chunk B가 길어질수록 ratio=0.15의 reduction이 50%+로 수렴. Insight 2 overlap도 같이 측정 (k = int(0.15 × len(chunk_B)), layer 1 vs L-1).

판단 기준:
- **Pass**: chunk B = 100 또는 200토큰에서 ratio=0.15 reduction ≥ 40%. 알고리즘이 정상이고 Phase 3 보고서의 해석이 맞다는 의미. Phase 4로 진행.
- **Fail**: chunk B 길이를 늘려도 reduction이 30% 미만에 머물면 알고리즘에 미세한 결함 가능성. Stop & 사용자에게 보고. Phase 4로 넘어가지 않음.

이 측정 결과를 Phase 4 보고서의 "Long-chunk sanity" 섹션에 표로 기록.

`@pytest.mark.requires_model and @pytest.mark.slow`로 마킹된 별도 pytest 테스트로도 추가하되 default CI에서는 skip.

## Step 3 — Phase 4 본 작업: Pipelining

읽어야 할 파일: `tasks/phase-4-pipelining.md`, `docs/paper-summary.md` (§5 LoadingController), `docs/design-decisions.md` (Phase 2의 KVStore disk backend skeleton).

### 핵심 원칙 (논문 §5)

KV 로딩 (디스크 → CPU/GPU) 과 selective recompute를 파이프라이닝해 TTFT에 추가 비용을 숨긴다. 정확성은 Phase 3 그대로 유지 — pipelining은 순수 성능 최적화여야 한다.

### 구현

`src/cacheblend/kv_store.py` 확장:
- `get_async(chunk_hash, layer_idx) -> Future`: ThreadPoolExecutor 기반. 디스크 backend 활성화 시 `pickle.load`을 worker thread에서.
- `prefetch_chunk(chunk_hash)`: 모든 layer의 KV 로딩을 큐에 등록.
- Phase 2 skeleton의 `disk_dir` 경로 사용. 디스크 backend 1차 실제 사용처.

`src/cacheblend/controller.py` 신규:
- `StorageProfile` dataclass (name, throughput_gbps, cost_per_gb).
- `LoadingController`:
  - `__init__(model, storage_profiles)`: model의 prefill_per_token 비용을 offline profile (1회 측정).
  - `estimate_recompute_delay(ratio, num_tokens)`: r% × prefill_full(num_tokens).
  - `estimate_load_delay(num_tokens, storage)`: per_token_kv_size × num_tokens / throughput.
  - `pick_recompute_ratio(num_tokens, storage, min_ratio=0.15) -> float`: T_recompute(r) ≈ T_load 가 되도록 r 선택, min_ratio 하한 적용. **추가 제약**: ratio가 0.5 근처에서 L2가 거의 0으로 수렴하므로 max_ratio=0.5 cap 적용 (Phase 3 Open Q 2 반영).

`src/cacheblend/fusor.py::fuse_selective_pipelined`:
- 기존 `fuse_selective`을 wrapping. layer i 진입 직전 layer i+1 KV의 prefetch 시작. layer 호출 직전 sync.
- ThreadPoolExecutor + Future.result() 패턴.
- Logits은 `fuse_selective`와 정확히 일치해야 함 (정확성 회귀 금지).

`benchmarks/ttft.py` 구현:
- `measure_ttft(method: Callable, request: dict, n_warmup=2, n_runs=10) -> dict`: median, p50, p95.
- GPU 환경에서는 `torch.cuda.synchronize()` 전후로 timing.
- CPU 환경에서는 `time.perf_counter()` 그대로.

### 테스트 (`tests/test_pipeline.py`)

- `test_pipelined_logits_match_unpipelined`: 동일 입력에서 `fuse_selective_pipelined`와 `fuse_selective`의 logit이 정확히 일치 (max_diff < 1e-5).
- `test_loading_controller_picks_sensible_ratio`: RAM(빠름)에서는 min_ratio = 0.15 hit. 가짜 슬로우 디스크(1Gbps)에서는 min_ratio보다 큰 값.
- `test_pipelined_ttft_lower_with_disk_kv`: 디스크 KV 시나리오에서 pipelined CacheBlend의 median TTFT가 full recompute보다 작음. CPU 환경에서는 가짜 sleep으로 디스크 latency 시뮬레이션 가능 — vast.ai GPU가 없어도 알고리즘 검증은 가능. Mac CPU + 가짜 디스크로 충분.

전부 `@pytest.mark.requires_model`. 짧은 입력 (S ≤ 30) + chunk B = 50토큰 1개 케이스 추가.

### vast.ai 측정은 이번 phase에서 하지 않는다

본 phase의 acceptance는 Mac CPU에서 검증 가능한 항목까지. Mistral-7B + 실제 NVMe 측정은 Phase 5와 함께 진행. 이유: vast.ai 비용/시간을 phase 4와 5에서 한 번에 묶어 진행하는 게 효율적이고, 본 phase의 본질(파이프라인 정확성, controller 로직)은 CPU + 가짜 디스크로도 검증 가능.

vast.ai 사용은 Phase 5에서. 단, `benchmarks/ttft.py`는 GPU 환경에서도 동작하도록 작성 (cuda.synchronize 분기 포함).

### 제약

- `@torch.compile` 금지. 새 의존성 금지 (ThreadPoolExecutor는 표준 라이브러리).
- Pipelining은 정확성 보존 — `fuse_selective` 결과와 logit max_diff < 1e-5.
- max_ratio = 0.5 cap을 controller에 명시 (Phase 3 sweep 결과 반영).
- 3번 시도 후 막히면 stop & 보고.

### 마무리

- `pytest -v -m "requires_model and not slow"` (Phase 1+2+3+4) 통과.
- `pytest -v -m "not gpu and not slow and not requires_model"` 통과.
- `python scripts/verify_phase.py --phase 4` 통과.
- `reports/phase-4-report.md` 작성. 명시 항목:
  * Long-chunk sanity 결과 (Step 2의 표) — pass/fail 명시
  * Pipelined vs unpipelined logit 일치 검증 (max_diff)
  * LoadingController가 디스크 throughput에 따라 추천하는 ratio 표 (RAM, fast SSD, slow disk 3종)
  * Mac CPU 가짜 디스크 시나리오에서의 TTFT 비교 (full recompute vs full reuse vs cacheblend pipelined)
  * vast.ai에서 측정할 항목 리스트 (Phase 5 위임)
  * GitHub PR URL
  * "Prompt archive: docs/prompts/phase-4-pipelining.md" cross-ref
- `python scripts/update_status.py --phase 4 --status completed`
- `git commit -m "Phase 4: pipelining + LoadingController + long-chunk sanity"` & `git push -u origin phase-4-pipelining`
- `gh pr create --repo chjs/cacheblend-hf --base main --head phase-4-pipelining --title "Phase 4: pipelining" --body-file reports/phase-4-report.md`
- `python scripts/send_report.py --phase 4`

시작 전 30초 stop-and-think: GOAL.md 재확인. "Phase 4의 본질은 정확성을 보존하면서 TTFT 추가 비용을 숨기는 것. 알고리즘은 Phase 3 그대로."
