# Phase 4 — Pipelining KV Loading and Recompute

## Objective

Selective recompute에 의한 추가 compute를, KV cache 로딩(특히 디스크에서)과 **병렬화** 하여 TTFT에 대한 추가 비용을 숨긴다.

## Inputs

- Phase 3의 `fuse_selective`
- 논문 §5 (LoadingController, pipelining)
- `docs/paper-summary.md` 의 시스템 섹션

## Steps

### 4.1 Async KV loading

`src/cacheblend/kv_store.py` 확장:

```python
class KVStore:
    def get_async(self, chunk_hash: str, layer_idx: int) -> Future[tuple[K, V]]:
        """
        Issue a load for one layer's KV. Returns immediately with a future.
        Backed by ThreadPoolExecutor (CPU-bound disk IO) or torch.cuda.Stream
        for GPU memcpy.
        """

    def prefetch_chunk(self, chunk_hash: str):
        """Start loading all layers' KV for a chunk into a queue."""
```

**Two implementation options**:
- **Option A (simpler)**: ThreadPoolExecutor, blocking `torch.load` on a worker thread.
- **Option B (faster)**: pinned memory + `cuda.Stream` non-blocking copy.

Default: Option A. Option B can be a follow-up if Phase 4 acceptance fails on TTFT.

### 4.2 Pipelined fuse_selective

`src/cacheblend/fusor.py` 의 `fuse_selective` 변형 (또는 `fuse_selective_pipelined`):

```python
def fuse_selective_pipelined(model, chunks, kv_store, ...):
    # Issue prefetch for layer 0 KV of all cached chunks
    # For layer i:
    #   wait for layer i KV to arrive
    #   START prefetch for layer i+1 KV (in background)
    #   selective recompute on layer i
    # ...
```

핵심: layer i 의 selective recompute 는 layer i+1 의 KV 로딩과 **시간적으로 겹친다**.

### 4.3 Loading Controller

`src/cacheblend/controller.py`:

```python
@dataclass
class StorageProfile:
    name: str               # "ram", "ssd", "hdd"
    throughput_gbps: float  # measured
    cost_per_gb: float      # for storage device selection

class LoadingController:
    def __init__(self, model: LayerwiseModel, storage_profiles: list[StorageProfile]):
        # Profile prefill_per_token cost offline
        ...

    def estimate_recompute_delay(self, ratio: float, num_tokens: int) -> float: ...
    def estimate_load_delay(self, num_tokens: int, storage: StorageProfile) -> float: ...

    def pick_recompute_ratio(
        self,
        num_tokens: int,
        storage: StorageProfile,
        min_ratio: float = 0.15,
    ) -> float:
        """
        Find ratio r such that recompute_delay(r) ≈ load_delay.
        Cap at min_ratio to preserve quality.
        """
```

수식 (논문 §5.1):
- `T_recompute(r, L) = r × Prefill_full(L)` — `Prefill_full` 은 offline profiled.
- `T_load(L, dev) = (per_token_kv_size × L) / throughput(dev)`
- `r* = max(r where T_recompute(r) = T_load, min_ratio)`

### 4.4 TTFT measurement

`benchmarks/ttft.py`:

```python
def measure_ttft(method: Callable, request: dict, n_warmup=2, n_runs=10):
    """method(request) -> first_token. Returns dict with median, p50, p95 TTFT."""
```

방법 비교:
1. Full recompute (HF baseline)
2. Prefix caching only (cache only chunk_1)
3. Full reuse
4. Selective recompute (no pipelining)
5. **Selective recompute + pipelining** ← our contribution at this phase

### 4.5 Tests

`tests/test_pipeline.py`:

```python
def test_pipelined_logits_match_unpipelined():
    """Pipelining is a perf optimization; logits must be identical."""

def test_pipelined_ttft_lower_than_full_recompute():
    """On synthetic input with disk-stored KVs, pipelined CacheBlend has
    lower TTFT than full recompute."""

def test_loading_controller_picks_sensible_ratio():
    """For RAM storage (fast load), ratio should hit min_ratio.
    For slow disk, ratio can grow up to where compute matches load."""
```

## Acceptance criteria

- [ ] Async KV loading 동작 (with `ThreadPoolExecutor` 최소)
- [ ] `fuse_selective_pipelined` 구현, logits이 비파이프라인 버전과 일치
- [ ] `LoadingController.pick_recompute_ratio` 구현 및 테스트
- [ ] `benchmarks/ttft.py` 로 5가지 방법의 TTFT 측정 가능
- [ ] 디스크 KV 시나리오에서 pipelined CacheBlend의 TTFT < full recompute의 TTFT
- [ ] `python scripts/verify_phase.py --phase 4` 통과

## Report

- TTFT 비교 표 (위 5가지 방법 × 1~3개 모델)
- pipelining의 wall-clock 이득 (μs 단위)
- LoadingController의 자동 선택 결과 (어떤 시나리오에서 어떤 ratio를 골랐는지)
- 실패 케이스가 있다면 (예: GPU 모델이 너무 작아 recompute가 너무 빨라 pipelining 이득 없음)

## Common pitfalls

1. **GIL**: Python ThreadPoolExecutor는 GIL 때문에 CPU-bound 작업은 가속이 안 된다. 디스크 IO는 GIL을 놓으므로 OK.
2. **CUDA Stream과 sync**: 만약 Option B (cuda.Stream)을 쓴다면, 다음 layer의 prefill 시작 전 그 layer의 KV가 GPU에 도달했음을 sync 해야 한다.
3. **Profile noise**: TTFT 측정은 first run이 항상 느리다 (CUDA warm-up, allocator). Warm-up 2-3 run 후 측정.
4. **Storage 가짜 디스크**: 진짜 NVMe 없는 환경에서는 `tmpfs` 에 저장 후 `time.sleep(...)` 으로 throughput을 흉내내는 mock을 쓸 수 있다.
