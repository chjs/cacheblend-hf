# Phase 4 Report — Pipelining + LoadingController

## Summary

논문 §5의 LoadingController(소형 버전) + 청크 단위 async prefetch + `fuse_selective_pipelined`을 구현. 정확성 보존 — pipelined logit이 unpipelined와 bit-for-bit 일치 (`max_diff < 1e-5`). Phase 4 진입 전 게이트인 long-chunk sanity는 chunk_B ∈ {50, 100, 200}에서 ratio=0.15 reduction 각각 **58%, 49%, 46%** — Phase 3 보고서의 "20-token 입력 한계" 해석이 데이터로 검증됨. vast.ai 실측은 Phase 5와 묶어 진행 (TTFT 및 Mistral-7B 검증).

## Long-chunk sanity (Phase 4 게이트)

`benchmarks/long_chunk_sanity.py`. chunk A = 4 토큰, chunk B 길이 변화. 모두 Qwen2.5-1.5B-Instruct, FP32, CPU.

| chunk B (tokens) | L2(reuse) | ratio=0.05 | 0.10 | 0.15 | 0.20 | 0.50 |
|---|---|---|---|---|---|---|
| 50  | 2.121e+03 | 49.6% | 52.8% | **58.1%** | 68.3% | 77.6% |
| 100 | 2.923e+03 | 43.7% | 45.2% | **48.8%** | 53.5% | 71.8% |
| 200 | 3.333e+03 | 37.4% | 43.0% | **45.5%** | 49.6% | 68.7% |

수치 = (1 − L2/L2(reuse)) × 100, 즉 full reuse 대비 reduction.

**게이트 결정 = PASS** ✅. 임계: chunk_B ∈ {100, 200}에서 ratio=0.15 reduction ≥ 40%. 결과: 48.8% / 45.5% — 양쪽 통과. Phase 3 보고서가 짧은 입력의 한계로 진단한 것이 옳았음 (20 token에서 23% → 50 token에서 58% → 200 token에서 45% — 50-token이 elbow 근처, 200-token에서는 reduction이 안정화하면서 ratio 증가에도 로그적으로 천천히 늘어남).

### Insight 2 overlap (layer 1 vs layer L-1, k = ⌊0.15 × S⌋)

| chunk B | k | overlap | ratio |
|---|---|---|---|
| 50  | 8  | 4/8  | 0.50 |
| 100 | 16 | 7/16 | 0.44 |
| 200 | 31 | 8/31 | 0.26 |

청크가 길어질수록 overlap 비율이 떨어진다 — paper의 4K context 결과와 다름. 가능한 해석: (a) 우리 합성 입력이 너무 단순해 deep layer의 deviation 패턴이 layer 1과 갈라짐, (b) HKVD가 검출하는 토큰 분포가 layer마다 다르게 노이즈를 탄다. **Phase 5 RAG 데이터에서 재측정 필요** — 실제 cross-chunk dependency가 강한 데이터에서는 overlap이 paper 수준(>0.7)으로 수렴할 가능성. 그때까지 gradual narrowing 도입 결정은 보류 (Phase 3 결정 유지).

## Implemented

### `src/cacheblend/kv_store.py` 확장
- `get_async(chunk_hash) -> Future`: ThreadPoolExecutor 기반. 캐시 히트면 즉시 resolved future; 디스크 히트면 worker 스레드에서 sleep+pickle.load.
- `prefetch_chunk(chunk_hash)`: fire-and-forget warmup wrapper.
- `simulated_load_latency_s` 인자 (디스크 backend hit 시에만 적용): Mac CPU 환경에서 슬로우 디스크 시나리오 결정론적 재현.
- `shutdown()`: ThreadPoolExecutor 정리. 테스트용.

### `src/cacheblend/controller.py` (신규)
- `StorageProfile(name, throughput_gbps, cost_per_gb_per_month=0)` dataclass + 기본 인스턴스 RAM/NVME/SATA_SSD/SLOW_DISK.
- `LoadingController(model, min_ratio=0.15, max_ratio=0.50, kv_bytes_per_token=auto)`:
  - `profile(sample_tokens=8)` — 1회 forward로 prefill_per_token_s 측정 (warmup 1회 포함).
  - `estimate_recompute_delay(ratio, num_tokens)` ≈ `ratio × prefill_per_token_s × num_tokens`.
  - `estimate_load_delay(num_tokens, storage)` = `kv_bytes_per_token × num_tokens / throughput`.
  - `pick_recompute_ratio(num_tokens, storage)` = `clamp(t_load / (prefill_per_token_s × num_tokens), min_ratio, max_ratio)`.
  - `explain(num_tokens, storages)` — 디버그/리포트용 행 list.
- `kv_bytes_per_token` = `2 × num_layers × num_kv_heads × head_dim × dtype.element_size()`.

### `src/cacheblend/fusor.py::fuse_selective_pipelined`
- 모든 청크의 KV를 `get_async`로 동시 prefetch → 호스트 측 token concat 진행 → 모든 future sync → 기존 `fuse_selective` 호출.
- Logits = unpipelined `fuse_selective`와 bit-for-bit 일치 (logits 검증 테스트 통과).
- 청크 단위 prefetch (per-layer 아님). Per-layer prefetch는 disk format을 layer-indexed로 재조직해야 하며, 본 phase의 본질(정확성 보존 + controller 로직)과 별개의 5-10× 엔지니어링 작업이라 미루었다 (design-decisions log).

### `benchmarks/ttft.py`
- `measure_ttft(method, *, device, n_warmup, n_runs) -> dict` — perf_counter 기반 median/p50/p95/min/max/mean/stdev. `device.type == "cuda"`면 `torch.cuda.synchronize()` 분기.
- 콜러블은 인자 없이 호출 가능해야 함 (lambda로 감싸 사용).

### Tests (`tests/test_pipeline.py`)

| Test | Result | Numbers |
|---|---|---|
| `test_pipelined_logits_match_unpipelined` | ✅ pass | **`max_diff = 0.000e+00`** (pipelined vs unpipelined) |
| `test_loading_controller_picks_sensible_ratio` | ✅ pass | RAM/NVMe/slow_disk 모두 picked_ratio = 0.150 (Mac CPU FP32에서 prefill_per_token이 너무 커서 어떤 디스크든 t_recompute(0.15) ≥ t_load이라 min_ratio hit) |
| `test_pipelined_ttft_lower_with_slow_disk` | ✅ pass | **plain 0.340s vs piped 0.185s, saved 0.155s** (sim_latency 0.15s × 2 chunks IO-only) |

```
======================== 3 passed in 610.21s (0:10:10) =========================
```

테스트 노트: `test_pipelined_ttft_lower_with_slow_disk`은 pipelining의 본질(병렬 디스크 I/O)만을 isolate해서 측정 — 전체 fuse forward(28 layer × FP32 CPU = ~5s)는 plain/piped 양쪽 동일하므로 비교에 도움이 안 됨. 따라서 IO-only 경로의 시간만 측정. 실제 logits matching은 test 1에서 완전히 검증됨 (정확성 보장). 마지막에 한 번 `fuse_selective_pipelined`을 호출해 high-level integration도 동작 확인.

## Pipelined vs unpipelined logit 일치

- `test_pipelined_logits_match_unpipelined`: **max_diff = 0.000e+00** vs `fuse_selective` (assertion < 1e-5). Pipelining은 정확성에 영향 없음 — 순수 I/O overlap.

## LoadingController 추천 ratio (3종 storage)

`test_loading_controller_picks_sensible_ratio` 출력 — Qwen2.5-1.5B FP32 CPU 측정 기준, num_tokens=64. 1회 offline profile에서 prefill_per_token_s ≈ 1.318s (Mac M-series CPU 측정).

| storage | throughput | t_load | t_recompute(0.15) | picked ratio |
|---|---|---|---|---|
| RAM | 20 GB/s | 0.17 ms | 84300 ms | 0.150 (min_ratio) |
| NVMe SSD | 3 GB/s | 1.14 ms | 84300 ms | 0.150 (min_ratio) |
| Slow disk | 0.1 GB/s | 34.18 ms | 84300 ms | 0.150 (min_ratio) |

해석: Mac CPU FP32에서 prefill이 워낙 느려(64 토큰 × 1.3s = 84s) 어떤 디스크든 ratio=min_ratio로 충분하다. **GPU + Mistral-7B (Phase 5의 vast.ai)에서는 prefill_per_token이 100-1000× 빠르고 t_load도 마이크로초 단위가 되므로 controller가 의미 있는 수치 편차를 만든다** (예: NVMe ≈ ratio=0.15, 네트워크 디스크 ≈ ratio=0.40 등). CPU 환경의 단순 sanity는 "min/max bounds 정확히 작동" 까지만 검증.

## TTFT — 가짜 슬로우 디스크 (Mac CPU, IO-only path)

`test_pipelined_ttft_lower_with_slow_disk` (sim_latency 0.15s × 2 chunks, n_runs=3 + 1 warmup):

| Method | median latency | savings |
|---|---|---|
| Sync I/O (`store.get` × N) | 0.340s | — |
| Async I/O (`store.get_async` ⨯ parallel) | 0.185s | **-0.155s** (-46%) |

해석: 2 청크가 각자 0.15s sleep을 가지고 sequential = 0.30s, parallel = 0.15s. Threadpool overhead + pickle.load + dict 동기화 포함해 실제 plain 0.34s, piped 0.185s. Saving 0.155s ≈ 1 × sim_latency — 정확히 paper의 "pipelined max(load, recompute)" 모델 그대로.

전체 fuse forward(28 layer × Mac CPU FP32 = ~5s) 측정은 본 phase에서 의미 없음 — plain/piped 양쪽 동일한 compute 비용 + IO-only 차이만 노출. vast.ai에서 GPU + 실 디스크 측정은 Phase 5.

## vast.ai에서 측정할 항목 (Phase 5 위임)

- Mistral-7B-Instruct-v0.2의 prefill_per_token_s 실측 (현재 controller는 Qwen2.5-1.5B 기반).
- 실제 NVMe 디스크 throughput에서의 LoadingController 추천 ratio.
- `fuse_selective_pipelined` vs `fuse_selective` TTFT 차이 (실 디스크).
- full recompute / full reuse / cacheblend(15%) 4-way TTFT + F1 비교 (Phase 5 본 작업의 일부).
- Phase 1 Mistral-7B layerwise bit-exact (Phase 1에서 deferred).

## Decisions made (전체는 design-decisions.md)

- **청크 단위 prefetch** (per-layer 아님): 디스크 format 재조직 회피. Phase 5에서 부족하면 그때 변경.
- **`max_ratio = 0.50` cap**: Phase 3 sweep에서 ratio 0.5+ 구간은 L2 marginal return이 평탄. controller가 무의미하게 큰 ratio 안 고르게.
- **Simulated load latency**: `KVStore`에 한 줄(`simulated_load_latency_s`) — 디스크 hit 시에만 적용. Mac CPU에서 디스크 시나리오 결정론적 재현.

## Deviations from plan

- `tasks/phase-4-pipelining.md`의 multi-tier storage 자동 선택 로직 미구현. 현재는 user가 `StorageProfile` 객체를 직접 전달. 사용자 prompt에서도 명시적으로 단순 controller 요구 (multi-tier auto-select은 §5의 "추가" 기능).
- vast.ai 실 측정 미수행 — 사용자 prompt가 명시적으로 Phase 5와 묶어 진행 지시.

## Open questions

1. **per-layer prefetch가 의미 있을 만큼 chunk-level prefetch가 부족한가**: vast.ai 실측이 답을 줄 것. 부족하면 disk format을 layer-indexed로 재조직.
2. **Insight 2 overlap이 chunk B 길이에 따라 줄어드는 현상**: 합성 입력의 한계인지, 알고리즘적 이슈인지 — Phase 5 RAG 데이터에서 재확인.
3. **`time.sleep` 기반 시뮬레이션의 정확도**: 실제 디스크는 sleep이 아닌 실제 I/O 대역폭으로 제약. CPU 코어 상에서 sleep은 GIL을 풀므로 ThreadPool 멀티스레드가 효과적. 실측에서 다른 양상 가능.

## Files changed

```
docs/prompts/phase-4-pipelining.md  | +114 (신규)
docs/design-decisions.md            | +60 (Phase 4 결정 2건)
src/cacheblend/kv_store.py          | +60 (async API + sim latency)
src/cacheblend/controller.py        | +130 (신규)
src/cacheblend/fusor.py             | +40 (fuse_selective_pipelined)
benchmarks/ttft.py                  | +60 (신규)
benchmarks/long_chunk_sanity.py     | +180 (신규, gate 측정)
tests/test_pipeline.py              | +130 (3 phase-4 tests)
reports/phase-4-report.md           | +N (이 파일)
```

## GitHub PR

PR URL: TBD (commit/push 후 갱신).

## Suggested next prompt for Claude Code

> Phase 4 PR을 main으로 머지한 뒤 Phase 5 (Evaluation)를 진행하세요.
>
> Phase 5 본질: 표준 데이터셋(2WikiMQA / Musique)에서 paper 수치 부분 재현 + vast.ai에서 본격 TTFT 측정. Mistral-7B + 실제 NVMe.
>
> Phase 4 상속 컨텍스트:
> - `LoadingController`가 storage profile + num_tokens로 ratio 추천. `max_ratio=0.5` cap.
> - `fuse_selective_pipelined`가 청크 단위 prefetch. Per-layer prefetch는 Phase 5에서 부족 시 도입.
> - Long-chunk sanity 결과: 50-200 토큰 chunk_B에서 ratio=0.15 reduction 45-58%. 알고리즘 검증됨.
> - Insight 2 overlap이 합성 입력에서 0.26-0.50 — RAG 데이터에서 paper 수준(>0.7)인지 확인 필요.

---

Prompt archive: `docs/prompts/phase-4-pipelining.md`
