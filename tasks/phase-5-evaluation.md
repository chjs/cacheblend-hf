# Phase 5 — End-to-End Evaluation

## Objective

표준 데이터셋에서 논문 Figure 12의 결과를 부분적으로라도 재현한다. **이전 phase 들의 합성 데이터 검증을 넘어서, 실제 RAG 워크로드에서 동작함을 보인다.**

## Inputs

- 모든 이전 phase 산출물
- `REFERENCES.md` 의 데이터셋 링크
- 논문 §7 (Evaluation)

## Steps

### 5.1 Dataset loaders

`benchmarks/datasets/`:

```
loaders/
├── musique.py       # load Musique dataset
├── twiki.py         # load 2WikiMQA
├── samsum.py        # SAMSum
└── multi_news.py    # MultiNews
```

각 loader는 동일 인터페이스:
```python
def load(split: str = "test", limit: int = None) -> list[dict]:
    """Each item: {'id', 'system', 'documents', 'query', 'answer'}"""
```

**최소 요구**: Musique 1개. Phase 5 에서 다 안 되면 다른 데이터셋은 follow-up.

### 5.2 Metrics

`benchmarks/metrics/`:

```python
def f1_score(pred: str, gold: str) -> float:
    """Token-level F1 (paper convention)."""

def rouge_l(pred: str, gold: str) -> float:
    """ROUGE-L F1 score."""

def aggregate(scores: list[float]) -> dict:
    """{'mean', 'std', 'p50', 'p95'}"""
```

데이터셋별 metric:
- Musique, 2WikiMQA → F1
- SAMSum, MultiNews → Rouge-L

### 5.3 Benchmark runner

`benchmarks/run_benchmark.py`:

```python
def run(
    model_id: str,
    dataset: str,
    method: str,                # 'full_recompute' | 'prefix' | 'full_reuse' | 'cacheblend'
    recompute_ratio: float = 0.15,
    limit: int = None,
    storage: str = "ram",       # 'ram' | 'disk'
) -> dict:
    """
    Returns: {
        'method': ..., 'dataset': ..., 'model': ...,
        'metric_name': 'f1' | 'rouge_l',
        'metric_value': float,
        'ttft_median_ms': float,
        'ttft_p95_ms': float,
        'kv_hit_rate': float,
    }
    """
```

CLI:
```bash
python -m benchmarks.run_benchmark \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset musique \
    --method cacheblend \
    --ratio 0.15 \
    --limit 50 \
    --storage disk \
    --output results/musique_cacheblend.json
```

### 5.4 Comparison plot

`benchmarks/plot_results.py`:

논문 Figure 12 스타일의 산점도 (TTFT vs F1):
- x축: TTFT (s)
- y축: F1 (or Rouge-L)
- 점들: 4 methods × dataset 별 1개

### 5.5 RAG-style chunking

논문은 SentenceTransformers 임베딩 + 512-token 청크 + top-k retrieval. 우리도 비슷하게:

```python
# benchmarks/rag.py
def build_rag_input(
    query: str,
    document_pool: list[str],
    embedder,                # sentence_transformers.SentenceTransformer
    chunk_size: int = 512,
    top_k: int = 6,
) -> dict:
    """Returns {'system', 'documents', 'query'} ready for fuse_*."""
```

### 5.6 Tests

`tests/test_e2e.py`:

```python
@pytest.mark.slow
def test_e2e_musique_small():
    """Run all 4 methods on 10 Musique examples. Verify:
    - cacheblend's F1 ≥ full_reuse's F1
    - cacheblend's F1 within 0.05 of full_recompute's F1
    """

@pytest.mark.slow
def test_e2e_ttft_speedup():
    """Disk-stored KVs: cacheblend's TTFT ≤ 0.7 × full_recompute's TTFT (10 examples)."""
```

`@pytest.mark.slow` — 일반 `pytest` 에서는 skip, `pytest -m slow` 로만 실행.

## Acceptance criteria

- [ ] 적어도 1개 데이터셋 (Musique 권장) 동작
- [ ] 4개 method × 1개 dataset × 50 sample 결과 표
- [ ] CacheBlend의 F1 ≥ full_reuse의 F1 + 0.05
- [ ] CacheBlend의 F1이 full_recompute의 F1과 0.05 이내
- [ ] (Optional, GPU 환경) TTFT 1.5× 이상 단축
- [ ] `python scripts/verify_phase.py --phase 5` 통과

## Report

- 데이터셋 별 4-way 비교 표 (F1/Rouge, TTFT)
- 논문 Figure 12 vs 우리 결과의 비교 토론
- 차이가 있다면 가능한 원인 (모델 크기, ratio, 청크 크기 등)
- Limitations 섹션: 우리 구현이 논문 대비 부족한 부분
- (있다면) 향후 개선 제안

## Common pitfalls

1. **Tokenizer mismatch with retrieved chunks**: 데이터셋의 청크가 모델 토크나이저와 다른 토크나이저로 잘렸을 수 있음. 우리 토크나이저로 다시 자른다.
2. **Answer format**: Musique/2WikiMQA의 정답은 짧다 (≤5 단어). 모델이 길게 답하면 F1이 낮게 나온다. 논문처럼 prompt에 "Answer within 5 words." 추가 권장.
3. **Sampling vs greedy**: 평가는 greedy decoding 으로 (재현성).
4. **TTFT 측정 정확성**: GPU sync 누락 시 측정값이 부정확.
5. **데이터셋 다운로드 실패**: HuggingFace datasets는 네트워크 필요. 미리 캐시.
