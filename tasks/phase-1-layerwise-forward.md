# Phase 1 — Layerwise Forward in HF Transformers

## Objective

HF transformers의 `forward`를 **layer 단위로 호출**할 수 있는 래퍼를 만든다. CacheBlend의 모든 후속 phase가 이 래퍼 위에 빌드되므로, 표준 forward와 **bit-exact (FP32) / 1e-3 (FP16)** 으로 일치해야 한다.

## Why this is critical

CacheBlend의 selective recompute는 layer마다 다른 토큰 부분집합에 대해 prefill을 수행한다. 이는 HF transformers의 일체형 `forward`로는 불가능하다. 우리는 다음 세 가지 호출이 가능해야 한다:

```python
hidden = model.embed_tokens(input_ids)
for i in range(num_layers):
    hidden, kv_i = model.prefill_layer(i, hidden, position_ids, past_kv=...)
logits = model.final_norm_and_lm_head(hidden)
```

그리고 이 시퀀스의 logits이 `model(input_ids).logits` 와 동일해야 한다.

## Inputs to read

- `ARCHITECTURE.md` 의 `model.py` 섹션
- HF transformers 소스: `src/transformers/models/<arch>/modeling_<arch>.py`
  - 시작 모델: Mistral 권장 (간단한 LlamaDecoderLayer 구조)
  - Decoder layer의 `forward(hidden_states, position_ids, past_key_value, ...)` 시그니처

## Steps

### 1.1 LayerwiseModel 구현

`src/cacheblend/model.py`:

```python
class LayerwiseModel:
    def __init__(self, hf_model_or_id: str | PreTrainedModel,
                 dtype=torch.float16, device="cuda"):
        # Load model, extract layers, embed, norm, lm_head
        ...

    @property
    def num_layers(self) -> int: ...
    @property
    def num_kv_heads(self) -> int: ...
    @property
    def head_dim(self) -> int: ...
    @property
    def rope_theta(self) -> float: ...

    def embed_tokens(self, input_ids: Tensor) -> Tensor:
        """input_ids: (B, S) -> hidden: (B, S, hidden)"""
        ...

    def prefill_layer(
        self,
        layer_idx: int,
        hidden_states: Tensor,           # (B, S, H)
        position_ids: Tensor,            # (B, S)
        past_key_value=None,             # for decode; None for full prefill
        attention_mask: Tensor = None,   # (B, S) or causal handled internally
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Returns (new_hidden, (k, v)) where k,v shape: (B, num_kv_heads, S, head_dim)"""
        ...

    def final_norm_and_lm_head(self, hidden: Tensor) -> Tensor:
        """hidden: (B, S, H) -> logits: (B, S, vocab)"""
        ...
```

**구현 노트**:
- HF의 `LlamaDecoderLayer` (Mistral도 같은 구조 사용)는 `forward(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, cache_position, position_embeddings)` 류의 시그니처를 가진다. 정확한 시그니처는 사용 중인 transformers 버전에 따라 다르므로, 실제 코드를 확인하고 맞춰야 한다.
- `cache_position`, `position_embeddings` 등 새로운 인자가 있다면 모두 정확히 전달해야 한다.
- attention mask는 causal mask가 자동으로 적용되도록 `None` 으로 넘기는 게 일반적이지만, 버전에 따라 명시적 mask 필요할 수 있음.
- KV는 `(B, num_kv_heads, S, head_dim)` 형태로 반환받는다 (GQA 모델의 경우 num_kv_heads ≠ num_heads).

### 1.2 Verification: bit-exact equivalence

`tests/test_layerwise.py`:

```python
@pytest.mark.parametrize("model_id", [
    "Qwen/Qwen2.5-1.5B-Instruct",
    # "mistralai/Mistral-7B-Instruct-v0.2",  # optional, slower
])
def test_layerwise_matches_standard(model_id):
    # Load with HF
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

    # Wrap
    lw = LayerwiseModel(hf_model, dtype=torch.float32, device="cpu")

    # Standard forward
    inputs = tokenizer("The quick brown fox jumps over", return_tensors="pt")
    with torch.no_grad():
        std_logits = hf_model(**inputs).logits

    # Layerwise forward
    with torch.no_grad():
        hidden = lw.embed_tokens(inputs.input_ids)
        position_ids = torch.arange(inputs.input_ids.shape[1])[None, :]
        for i in range(lw.num_layers):
            hidden, _ = lw.prefill_layer(i, hidden, position_ids)
        lw_logits = lw.final_norm_and_lm_head(hidden)

    # Compare
    max_diff = (std_logits - lw_logits).abs().max().item()
    assert max_diff < 1e-5, f"Logit mismatch: {max_diff}"
```

**중요**: FP32 권장 (FP16/BF16에서는 누적 오차로 1e-3 수준의 차이가 발생할 수 있음).

### 1.3 Position-aware test (KV 저장/복원의 사전 단계)

```python
def test_kv_extraction(model_id):
    """Verify that the KV we extract per layer is what HF would have stored."""
    # Use HF with use_cache=True, get past_key_values
    # Extract KV with our LayerwiseModel
    # They should match exactly
    ...
```

### 1.4 (Optional) Bigger model smoke test

`tests/test_layerwise.py::test_mistral_smoke`: Mistral-7B에서도 layerwise == standard. GPU 가용 시에만 실행 (`@pytest.mark.gpu`).

## Acceptance criteria

- [ ] `LayerwiseModel` 의 모든 메서드 구현
- [ ] `tests/test_layerwise.py::test_layerwise_matches_standard` 통과 (Qwen2.5-1.5B FP32에서 max_diff < 1e-5)
- [ ] `tests/test_layerwise.py::test_kv_extraction` 통과
- [ ] (Optional) Mistral-7B 검증 통과 (GPU 환경에서)
- [ ] `python scripts/verify_phase.py --phase 1` 통과

## Report

다음을 포함한다:
- 사용한 transformers 버전
- 어떤 모델에서 검증했고 max logit diff가 얼마였는지
- HF API의 어떤 부분이 까다로웠는지 (특히 `position_embeddings`, `cache_position` 등)
- LayerwiseModel이 다양한 architecture에서 동작하도록 일반화되어 있는지, 아니면 특정 모델 family에 한정인지
- Phase 2 시작 전 알아야 할 것

## Common pitfalls (먼저 알고 시작하세요)

1. **transformers 버전 차이**: 버전에 따라 `LlamaDecoderLayer.forward` 시그니처가 다르다. 우리 `requirements.txt` 에 핀된 버전 기준으로 작업.
2. **causal mask**: HF는 보통 내부에서 causal mask를 만든다. 직접 만들어 넘기면 형태가 안 맞을 수 있다.
3. **`position_embeddings` 인자**: 최신 transformers는 RoPE를 layer 진입 전에 미리 계산해 `position_embeddings = (cos, sin)` 으로 넘긴다. 이걸 누락하면 silent하게 잘못된 결과가 나온다.
4. **GQA**: K, V의 head 개수가 Q와 다르다. shape 가정 시 주의.
5. **bf16 vs fp32**: 검증은 FP32로. fp16/bf16에서는 acceptable tolerance를 1e-3으로 풀어줄 것.
