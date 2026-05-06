# 📍 Phase Roadmap

이 파일은 전체 로드맵의 한눈 요약입니다. 각 phase의 상세 작업은 `tasks/phase-N-*.md`를 참조하세요.

---

## Phase 0 — Setup & Analysis

**Objective**: 작업 환경을 갖추고, 논문/LMCache 코드 분석 산출물을 만든다.

**Deliverables**:
- `requirements.txt` 동작 확인 (`pip install -r requirements.txt && python -c "import torch, transformers"`)
- `docs/paper-summary.md` 채움 (이미 초안 있음 — 보강만)
- `docs/lmcache-analysis.md` 채움 — LMCache의 CacheBlend 부분만 발췌해서 우리 구현이 어떻게 단순화될지 설계
- `.env` 셋업 가이드 동작 확인 (이메일 발송 dry-run 통과)

**Acceptance**:
- `python scripts/send_report.py --phase 0 --dry-run` 통과
- 분석 문서 2개에 빈 섹션 없음

📄 자세한 내용: `tasks/phase-0-analysis.md`

---

## Phase 1 — Layerwise Forward

**Objective**: HF transformers의 `forward`를 layer 단위로 호출 가능한 형태로 래핑한다. CacheBlend의 모든 후속 phase가 여기에 의존한다.

**Deliverables**:
- `src/cacheblend/model.py`: `LayerwiseModel` 클래스
  - `prefill_layer(layer_idx, hidden_states, position_ids, kv_cache=None)` → `(new_hidden, new_kv)`
  - `embed_tokens(input_ids)` → `hidden_states`
  - `final_norm_and_lm_head(hidden_states)` → `logits`
- 표준 `model(input_ids).logits`와 layerwise 호출 결과가 **bit-exact (FP32) / 1e-3 (FP16)** 일치

**Acceptance**:
- `tests/test_layerwise.py::test_layerwise_matches_standard` 통과
- 최소 2개 모델에서 검증 (작은 모델 + Mistral-7B 또는 등가)

📄 자세한 내용: `tasks/phase-1-layerwise-forward.md`

---

## Phase 2 — KV Storage & Full Reuse with RoPE Recovery

**Objective**: 청크 단위 KV cache 저장/로드 시스템과, 위치 정보를 보정한 KV concatenation을 구현한다. 이 phase의 결과는 **Full KV reuse (논문의 PromptCache 방식)** 와 동등하다 — 즉, cross-attention은 아직 복원하지 않는다.

**Deliverables**:
- `src/cacheblend/kv_store.py`: 해시 기반 청크 KV 저장소 (in-memory + 옵션 disk)
- `src/cacheblend/rope.py`: KV cache의 위치 재인코딩 (RoPE 회전 행렬 곱)
- `src/cacheblend/fusor.py` 의 `full_reuse(...)` 함수
- 합성 데이터에서 PromptCache 수준의 품질 (논문 §3.3의 한계 재현 가능)

**Acceptance**:
- `tests/test_kv_reuse.py`: 동일 청크의 prefix/non-prefix 위치에서 KV가 RoPE 보정 후 정합
- Full reuse 결과가 full recompute와 logit이 다름은 OK (cross-attention 차이). 하지만 **동일 prefix 케이스에서는 일치**해야 함

📄 자세한 내용: `tasks/phase-2-kv-storage.md`

---

## Phase 3 — Selective KV Recompute (Core CacheBlend) ⭐

**Objective**: 논문의 핵심. HKVD 토큰을 식별하고, gradual filtering으로 layer마다 점진적으로 좁혀가며 선택적으로 재계산한다.

**Deliverables**:
- `src/cacheblend/hkvd.py`: KV deviation 계산, top-k 토큰 선택, gradual filtering
- `src/cacheblend/fusor.py` 의 `selective_recompute(...)` 함수
- `recompute_ratio` 를 인자로 받아 layer별 ratio schedule 결정 (논문: r1 > r2 > ... slightly decreasing)

**Acceptance**:
- `tests/test_selective.py`: 합성 다중 청크 입력에서 selective recompute 결과가 full reuse보다 full recompute에 더 가깝다 (logit L2 거리 기준)
- recompute_ratio=15% 에서 logit 차이가 full reuse 대비 **유의미하게 감소** (구체 수치는 task 파일에 명시)

📄 자세한 내용: `tasks/phase-3-selective-recompute.md`

---

## Phase 4 — Pipelining

**Objective**: KV 로딩과 selective recompute를 파이프라이닝하여 TTFT 추가 비용을 숨긴다.

**Deliverables**:
- `src/cacheblend/controller.py`: `LoadingController` (recompute ratio 자동 결정, 저장 디바이스 추천)
- async 또는 별도 스레드/스트림 기반 KV prefetch
- TTFT 측정 인프라 (`benchmarks/ttft.py`)

**Acceptance**:
- `tests/test_pipeline.py`: 디스크 KV 로딩 시나리오에서, selective recompute에 의한 TTFT 증가가 baseline 대비 작음을 측정
- 정확성 회귀 없음 (Phase 3 테스트 그대로 통과)

📄 자세한 내용: `tasks/phase-4-pipelining.md`

---

## Phase 5 — Evaluation

**Objective**: 표준 데이터셋에서 논문의 결과를 부분적으로라도 재현한다.

**Deliverables**:
- `benchmarks/run_benchmark.py`: full recompute / prefix caching / full reuse / CacheBlend 4-way 비교
- 데이터셋 로더: 2WikiMQA, Musique (최소 한 개 필수, 가능하면 둘 다)
- F1 / Rouge-L / TTFT 결과 표

**Acceptance**:
- CacheBlend의 F1이 full recompute 대비 0.02 이내
- CacheBlend의 TTFT가 full recompute 대비 1.5× 이상 단축 (단일 GPU, 디스크 KV)

📄 자세한 내용: `tasks/phase-5-evaluation.md`

---

## Status tracking

현재 phase 상태는 `reports/STATUS.md`에서 관리한다. 각 phase 완료 시 자동 갱신된다.
