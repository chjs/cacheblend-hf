# Phase 0 Report — Setup & Analysis

## Summary

작업 환경(Python 3.14.4, torch 2.11.0, transformers 4.57.6) 구동을 확인하고, 이메일 파이프라인 dry-run을 통과시켰으며, `docs/paper-summary.md`를 보강하고 `docs/lmcache-analysis.md`를 작성하여 LMCache의 CacheBlend 구현(`chjs/LMCache@fix/cacheblend-vllm-v0.17.1-compat`, commit `9f8aa4d`)을 우리 아키텍처와 매핑했다. 모든 필수 skeleton 파일이 존재하며 `cacheblend` 패키지는 editable install 후 정상 import된다.

## What was done

### Implemented / written

- `docs/paper-summary.md` — Figure 6 (recompute ratio vs forward attention deviation), Figure 8 (HKVD rank 상관), §5 LoadingController 알고리즘 섹션을 새로 추가. 기존 알고리즘/RoPE/숫자 섹션은 그대로 유지.
- `docs/lmcache-analysis.md` — 빈 템플릿이었던 파일을 모든 섹션 채워서 최종본으로 작성. LMCache 트리 구조, 5개 핵심 질문 답, LMCache↔우리 매핑 표, "복잡성 위협 5선", Phase 0 결정 5건의 decision log 포함.
- `external/LMCache/` — 사용자가 지정한 fork(`chjs/LMCache`, branch `fix/cacheblend-vllm-v0.17.1-compat`)를 shallow clone. 향후 Phase 1~3 작업 시 즉시 참조 가능.
- `cacheblend-hf` 패키지 editable install (`pip install -e .`) — `PYTHONPATH=src` 없이 `import cacheblend` 동작.

### Coding deltas

거의 없음 (Phase 0의 본질). 신규 코드 0줄, 신규 분석 문서 2개 (총 ~22KB), editable install 1회.

### Tests

| Check | Result | Notes |
|---|---|---|
| `python -c "import torch, transformers"` | ✅ pass | torch 2.11.0, transformers 4.57.6, Python 3.14.4 |
| `import cacheblend` (editable install) | ✅ pass | version 0.0.1 |
| `python scripts/send_report.py --phase 0 --dry-run` | ✅ pass | `.env`에 GMAIL 자격증명 미입력 상태 — dry-run은 통과 (warning), 실제 발송은 자격증명 필요 |
| `python scripts/verify_phase.py --phase 0` | ✅ pass | 3/3 파일 존재 + dry-run 통과 |

### Acceptance criteria checklist

`tasks/phase-0-analysis.md` 기준:

- [x] `python -c "import torch, transformers"` 성공
- [x] `python scripts/send_report.py --phase 0 --dry-run` 성공
- [x] `docs/paper-summary.md`에 빈 섹션 없음 (Fig 6/8/§5 보강)
- [x] `docs/lmcache-analysis.md`의 5가지 핵심 질문에 모두 답해짐 (HKVD 함수 위치 / layerwise prefill 위치 / RoPE shift 위치 / KV 저장 단위 / recompute_ratio scheduling)
- [x] skeleton 체크리스트 13개 파일 모두 존재

## LMCache 분석 — 5가지 핵심 답

1. **HKVD 선택 함수 위치**: `external/LMCache/lmcache/v1/compute/blend/blender.py`, `LMCBlender.process_qkv`, lines 88–113. `if layer_id in check_layers:` 블록 안에서 `diff_k = sum((k_fresh - k_cached)^2, dim=hidden)`로 squared L2를 계산하고 `torch.topk(diff_k, k=int(total_len * recomp_ratios[0]))`로 인덱스를 뽑은 뒤 `torch.sort`로 정렬한다.
2. **Layer-by-layer 부분 prefill 위치**: `lmcache/v1/compute/models/base.py`, `LMCBaseModel.compute_layer` (lines 67–142). `@torch.compile`된 generator로, 각 layer마다 `vllm_model.model.layers[idx]`의 input_layernorm → qkv_proj → `_process_qkv` (model-specific) → `blender.process_qkv` (HKVD select & K/V scatter) → `lmc_attn_layers[idx].forward_contiguous` (FlashAttn or sparse) → o_proj → MLP를 직접 풀어서 호출한다. vLLM hook이 아니라 vLLM 모델 객체의 attribute에 직접 접근하는 형태.
3. **RoPE shift 위치**: `lmcache/v1/compute/positional_encoding.py`. (a) `BasicReverseRope.reverse_encode`는 검증용 pure-PyTorch 경로, (b) `FusedRope.fused_encode(old_positions, new_positions, k)`가 hot path로 CUDA 커널 `lmc_ops.rotary_embedding_k_fused`을 한 번 호출해 `R^{-1}_{old}` → `R_{new}`를 결합 적용한다. `get_fused_rope`가 `rotary_dim==head_size`, `rope_scaling is None`, `partial_rotary_factor==1.0`을 강제 검증.
4. **KV 저장 단위**: chunk(가변 길이, separator string `blend_special_str` 기본 `" # # "`로 분할 → `SegmentTokenDatabase._fast_split_by_subtensor`)가 외부 단위, 그 안에서 token이 인덱싱 가능. Memory format은 blending이 켜지면 `MemoryFormat.KV_2TD`로 layer-per-chunk 객체가 `(2, T, D)` 형태(K/V 분리, token-major). Page는 vLLM 측 page table에서 관리.
5. **Recompute ratio scheduling**: **사실상 없음.** `LMCBlender.__init__`에서 `recomp_ratios: list[float]`을 받지만 `process_qkv`는 `recomp_ratios[0]`만 사용한다. 코드에 `TODO(Jiayi): support different ratios for different layers`와 `TODO(Jiayi): remove [0] hardcode` 명시. 즉 LMCache는 단일 ratio를 단일 check layer(default `[1]`)에서만 적용. 논문의 layer별 점진 감소(`r_1 > r_2 > … > r_target`)는 우리가 직접 설계해야 한다.

## Decisions made

`docs/lmcache-analysis.md` "Decisions log"와 `docs/design-decisions.md`로 동기화 권장(아직 후자에는 미반영, Phase 1 시작 시 사용자 확인 후 옮길 예정):

- **K 저장 convention**: pre-RoPE로 저장 (LMCache는 post-RoPE + fused inverse-rotation 커널 사용). 이유: 커널 의존성 제거 + Phase 2 RoPE 보정이 한 번의 forward 회전으로 끝남. 비용: HF arch 내부에서 `apply_rotary_pos_emb` *직전*에 K를 가로채야 함. Phase 1 bit-exact 테스트로 검증 시 막히면 재고.
- **per-layer recompute ratio 감소 schedule**: Phase 3에서 구현 보류. LMCache처럼 단일 ratio(논문의 15%)를 단일 check layer 이상에서 그대로 사용. Phase 5 F1이 0.02 budget을 넘으면 그때 `gradual_filter_schedule` 추가.
- **check layer**: 기본 `[1]` (LMCache default). 튜닝 안 함.
- **`@torch.compile` 사용 시점**: Phase 4 timing 측정 단계 전까지 사용 안 함. Bit-exact 테스트가 우선.
- **Sparse attention backend**: 도입 안 함. HF 표준 attention(SDPA/eager)을 masked slice 위에서 그대로 사용. Phase 4 TTFT가 critically 부족하면 그때 재고.

## Deviations from plan

- `tasks/phase-0-analysis.md`는 LMCache 본가(`https://github.com/LMCache/LMCache.git`)를 clone하라고 명시. 사용자가 지정한 fork(`chjs/LMCache`, branch `fix/cacheblend-vllm-v0.17.1-compat`)를 사용. 코드 구조는 동일하지만 vLLM v0.17.1 호환 패치가 들어간 dev merge 버전(commit `9f8aa4d`).
- `pip install -e .`을 추가로 실행해 editable install. Phase 0 task에는 명시되지 않았으나 향후 `tests/`에서 `import cacheblend`를 `PYTHONPATH` 없이 쓰려면 필수.

## Open questions / blockers

1. **이메일 발송 자격증명**: 현재 `.env`의 `GMAIL_ADDRESS`와 `GMAIL_APP_PASSWORD`가 비어 있음. dry-run은 통과하지만 `python scripts/send_report.py --phase 0` (실제 발송)은 SMTP 인증 단계에서 실패함. 사용자가 `.env`를 채워야 실제 메일이 나감.
2. **HF 모델에서 K를 pre-RoPE로 가로채는 방법**: 현재 결정은 "decoder layer 내부에서 `apply_rotary_pos_emb` 직전에 K를 캡처"인데, HF `LlamaAttention.forward` / `MistralAttention.forward`의 정확한 분기점을 Phase 1 시작 시 확인해야 함. forward attention output을 monkey-patch할지, custom forward를 wrap할지 결정 필요.
3. **Per-layer ratio decay 보류 여부 확인**: 위 Decisions에 명시했으나 사용자 동의 필요 — Phase 5에서 F1 부족 시 추가 작업으로 들어가는지, 아니면 처음부터 구현해두는지.

## Files changed

```
docs/paper-summary.md       | +28 (Fig 6, Fig 8, §5 LoadingController 섹션 추가)
docs/lmcache-analysis.md    | +180 (전부 신규 작성)
reports/phase-0-report.md   | +N (이 파일)
external/LMCache/           | clone (untracked)
src/cacheblend.egg-info/    | editable install 부산물 (gitignored 권장)
```

(실제 git 워크트리에서는 `git status`로 추적되지 않는 untracked 파일도 다수 — 본 프로젝트가 `git init` 안 된 상태로 보임. Phase 1 시작 전에 git 초기화 권장.)

## Next phase prep

Phase 1을 시작할 때 알아야 할 컨텍스트:

- `LayerwiseModel.prefill_layer`는 HF `<Arch>DecoderLayer.forward`를 그대로 호출하되, **`past_key_value`를 통해 (k, v)를 pre-RoPE로 추출**해야 함. 가장 깨끗한 구현은: HF가 expose하는 `output_attentions=True`나 `use_cache=True` 인터페이스를 신뢰하고, 추출한 KV가 post-RoPE라면 §1.3 결정대로 inverse-rotation을 storage 시점에 적용. 단순함 우선이라면 **post-RoPE로 일단 저장하고 Phase 2에서 fuse 시점에 inverse-rotation을 한 번 더 도입**해도 됨 — 이 trade-off는 Phase 1 코드 작성 직전 사용자와 확인.
- bit-exact test (`tests/test_layerwise.py::test_layerwise_matches_standard`)는 FP32에서 1e-5, FP16에서 1e-3 tolerance. Qwen2.5-1.5B-Instruct로 충분히 통과 가능 — 최소 2개 모델 검증 요건이 있으므로 Mistral-7B-Instruct-v0.2도 후속(GPU 인스턴스에서) 돌릴 준비.
- LMCache의 `LMCBaseModel.compute_layer`는 vLLM 객체에 직접 접근하므로 그대로 베끼면 안 된다. HF의 표준 `decoder_layer(...)`를 호출하는 형태로 다시 쓸 것.

## Suggested next prompt for Claude Code

> Phase 0 완료. 이제 Phase 1을 진행하세요.
>
> 다음 파일을 먼저 읽으세요: `tasks/phase-1-layerwise-forward.md`, `docs/paper-summary.md`, `docs/lmcache-analysis.md` (특히 "What's most threatening to our simplicity" 섹션과 Decisions log).
>
> Phase 0의 핵심 결정사항:
> - K는 **pre-RoPE**로 저장하기로 했음 — `LayerwiseModel.prefill_layer`가 K를 추출할 때 `apply_rotary_pos_emb` 직전 값을 잡아야 함. HF `LlamaAttention.forward` / `MistralAttention.forward` 구조를 먼저 확인하고, monkey-patch가 필요한지 판단할 것.
> - `@torch.compile`은 사용 금지 (bit-exact 테스트 안정성).
> - 모델은 일단 `Qwen/Qwen2.5-1.5B-Instruct` (FP32 CPU)로 검증, Mistral-7B는 vast.ai에서.
>
> 막히면 3번 시도 후 멈추고 보고.
