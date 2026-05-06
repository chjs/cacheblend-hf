# 🎯 GOAL — The Unchanging North Star

> **Read this file at the start of every session. If a step does not advance this goal, stop and re-plan.**

## What we are building

CacheBlend을 **HuggingFace Transformers** 위에 **최소한**으로 구현한다.

CacheBlend의 핵심 아이디어 (논문 §4):
1. 입력이 여러 텍스트 청크의 결합일 때, 각 청크의 KV cache를 **사전 계산**해 둔다.
2. 추론 시점에는 모든 청크의 KV를 **재사용**하되, **HKVD(High KV Deviation) 토큰**의 KV만 **선택적으로 재계산**(typically ≤15%)하여 cross-attention을 복원한다.
3. 결과: **full KV recompute의 품질** + **full KV reuse에 가까운 속도**.

## Target scenario (변경 금지)

```
LLM input = [system_prompt] + [doc_1] + [doc_2] + ... + [doc_N] + [user_query]
```

- 각 문서의 길이는 가변 (random)
- 문서의 개수는 가변 (random, e.g., 1~10)
- 문서의 순서는 가변 (random) — 즉, **prefix caching만으로는 부족하다**
- system_prompt는 항상 prefix
- user_query는 항상 suffix
- 동일 문서가 여러 요청에서 재사용된다 (KV cache reuse 기회)

## Success metrics

| 지표 | 목표 |
|---|---|
| KV cache reuse ratio | 모든 비-신규 청크에 대해 KV 재사용 |
| Selective recompute ratio | 평균 ~15% 토큰 (논문 default) |
| 품질 (F1 / Rouge-L) | full KV recompute 대비 **0.02 이내** 손실 |
| 정확성 검증 | Phase 1 layerwise forward는 standard forward와 logit이 **bit-exact** (또는 1e-5 tol 이내) 일치 |
| 코드 단순성 | LMCache 전체 의존성 없음. PyTorch + transformers + (optional) faiss 만으로 |

## Non-goals (해서는 안 되는 것)

❌ LMCache 전체 시스템 포팅 (분산 저장소, 다중 워커, scheduling 등)
❌ vLLM/SGLang 등 별도 추론 엔진 통합
❌ Quantization, Distserve, chunked prefill 같은 **직교적** 최적화
❌ Mamba/Griffin 등 non-transformer 모델
❌ 새로운 알고리즘 발명 — 우리는 **논문 §4를 충실히 구현**한다

## 작업 원칙

1. **단순함 우선**: HF transformers의 표준 기능을 최대한 그대로 쓰고, CacheBlend가 요구하는 만큼만 hook을 건다.
2. **검증 가능한 단계**: 각 Phase는 자체 테스트로 통과/실패가 명확해야 한다.
3. **목표 회귀**: 헷갈리면 이 파일을 다시 읽는다.
