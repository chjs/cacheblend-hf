# Phase 2 — KV Storage & Full Reuse with RoPE Recovery

## Objective

청크 단위 KV cache 저장/로드 시스템과, 위치 정보를 보정한 KV concatenation을 구현한다. 이 phase의 결과는 **Full KV reuse** 와 동등하다 — 즉, cross-attention은 아직 복원하지 않는다. (Cross-attention 복원은 Phase 3에서.)

이 phase가 정확히 동작해야 Phase 3의 selective recompute가 의미 있다.

## Inputs

- Phase 1의 `LayerwiseModel`
- `docs/paper-summary.md` 의 RoPE 섹션
- `REFERENCES.md` 의 RoFormer 논문 (필요 시)
- 논문 Appendix A (n차원 RoPE 위치 회복 증명)

## Steps

### 2.1 Chunker

`src/cacheblend/chunker.py`:

```python
@dataclass
class Chunk:
    text: str
    token_ids: Tensor       # (S_chunk,)
    position: int           # absolute start position in current input
    hash: str               # for KV store lookup
    is_cached: bool = False # whether KV was a cache hit

def chunk_input(
    system_prompt: str,
    documents: list[str],
    user_query: str,
    tokenizer,
) -> list[Chunk]:
    """
    Split input into chunks. Each document is one chunk.
    System prompt and user query are also chunks.
    Hashes are computed from chunk text only (position-independent).
    """
```

**중요**: 해시는 **텍스트만**으로 계산한다. 위치는 별도. 같은 문서가 다른 위치에 와도 같은 해시.

### 2.2 KVStore

`src/cacheblend/kv_store.py`:

```python
class KVStore:
    """
    In-memory store: dict[hash] -> List[Tuple[K, V]] of length num_layers.
    K, V shape: (num_kv_heads, S_chunk, head_dim) — batch dim collapsed.

    K is stored WITHOUT RoPE applied (i.e., pre-rotation).
    The Fusor will apply RoPE based on target position.
    """

    def __init__(self, max_chunks: int = 1024, disk_dir: str | None = None):
        ...

    def get(self, chunk_hash: str) -> Optional[list[tuple[Tensor, Tensor]]]: ...
    def put(self, chunk_hash: str, kv: list[tuple[Tensor, Tensor]]): ...
    def has(self, chunk_hash: str) -> bool: ...
    def evict_lru(self): ...

    # Disk backend (optional, gated by disk_dir)
    def _save_to_disk(self, chunk_hash, kv): ...
    def _load_from_disk(self, chunk_hash): ...
```

**디자인 결정**: K는 pre-RoPE 저장 (논문은 명시적으로 안 했지만 우리는 이렇게 함). 이유: 위치 보정을 곱셈 한 번으로 처리. 대안 (post-RoPE 저장 → un-rotate → re-rotate) 은 부동소수점 손실이 있다.

> 만약 HF의 `LlamaDecoderLayer` 가 RoPE를 attention 모듈 안에서 적용하고 K cache는 post-RoPE를 저장한다면, 우리는 그 cache를 가로채서 un-rotate해서 저장하거나, RoPE 적용 직전 hook을 걸어 pre-RoPE를 저장해야 한다. **Phase 1에서 이 점을 미리 확인할 것.**

### 2.3 KV Pre-computation

`src/cacheblend/precompute.py`:

```python
def precompute_chunk_kv(
    model: LayerwiseModel,
    chunk_text: str,
    tokenizer,
    use_dummy_prefix_for_position: bool = False,
) -> list[tuple[Tensor, Tensor]]:
    """
    Run a full prefill on the chunk text alone, return per-layer (K, V).
    K is returned PRE-RoPE (so the Fusor can rotate to any position).
    """
```

**노트**: PromptCache는 dummy prefix를 prepend해서 위치를 시뮬레이션하지만, 우리는 RoPE 회전 행렬을 직접 적용하므로 dummy prefix 불필요. 우리 방식이 더 정확하고 빠르다.

### 2.4 RoPE position shift

`src/cacheblend/rope.py`:

```python
def apply_rope_shift(
    k_pre_rope: Tensor,      # (..., S, head_dim) — pre-RoPE K
    target_positions: Tensor,# (S,) target absolute positions
    rope_theta: float,
    head_dim: int,
) -> Tensor:
    """
    Apply RoPE rotation matrix R_{Θ, m} for each m in target_positions.
    Returns post-RoPE K.
    """
    # Build cos/sin table
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2) / head_dim))
    freqs = target_positions.float()[:, None] * inv_freq[None, :]   # (S, head_dim/2)
    cos = freqs.cos()
    sin = freqs.sin()
    # Apply rotation. Two conventions exist (interleaved vs split-half).
    # Pick the same convention as the model in use.
    ...
```

**중요**: HF Llama/Mistral은 "split-half" RoPE convention을 쓴다 (rotate_half). 다른 모델은 interleaved convention을 쓸 수 있다. `LayerwiseModel` 에서 model의 RoPE 모듈을 추출해 그대로 쓰는 것이 안전하다.

### 2.5 Fusor — full reuse path

`src/cacheblend/fusor.py`:

```python
def fuse_full_reuse(
    model: LayerwiseModel,
    chunks: list[Chunk],
    kv_store: KVStore,
    tokenizer,
) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
    """
    For each chunk:
      - if cached: load KV, apply RoPE shift to K based on chunk.position
      - if not cached: full prefill (with current concatenated context as prefix)
                      OR (simpler) just run alone and shift like cached chunks
                      — the simpler choice is what PromptCache does and what we do here.
    Concatenate per-layer K, V across chunks.
    Run final layer + lm_head on the last few tokens to produce logits.

    Returns: (logits, fused_kv_cache)
    """
```

### 2.6 Tests

`tests/test_kv_reuse.py`:

```python
def test_rope_shift_correctness():
    """A cached chunk re-positioned via apply_rope_shift should match running
    the chunk through the model at that position from scratch."""
    # Take some text. Pre-compute KV alone (position 0..S).
    # Apply RoPE shift to position P..P+S.
    # Compare to: run "<P dummy tokens> <text>", extract K of text.
    # Should match (within FP tolerance).

def test_full_reuse_matches_full_recompute_when_only_one_chunk_at_prefix():
    """When the input is a single cached chunk + uncached query at the prefix,
    full reuse should be EXACTLY equivalent to full recompute."""

def test_full_reuse_diverges_with_multiple_chunks():
    """Reproduce the paper's observation: with multiple chunks, full reuse's
    forward attention diverges from full recompute. This is expected — Phase 3
    fixes it."""
```

세 번째 테스트는 "**기대된 발산**" 을 검증한다. divergence가 있어야 우리 진단이 옳다.

## Acceptance criteria

- [ ] `Chunker`, `KVStore`, `apply_rope_shift`, `precompute_chunk_kv`, `fuse_full_reuse` 모두 구현
- [ ] `test_rope_shift_correctness` 통과
- [ ] `test_full_reuse_matches_full_recompute_when_only_one_chunk_at_prefix` 통과
- [ ] `test_full_reuse_diverges_with_multiple_chunks` 통과 (logit L2 차이가 측정 가능한 수준)
- [ ] `python scripts/verify_phase.py --phase 2` 통과

## Report

- 어떤 RoPE convention을 사용했는지 (split-half vs interleaved)
- KV를 pre-RoPE로 저장하는 결정의 근거
- divergence 측정 결과 (logit L2 distance를 청크 수에 따라 그래프 또는 표로)
- Phase 3 시작 전 사용자가 결정해야 할 사항

## Common pitfalls

1. **Pre-RoPE vs post-RoPE**: HF의 `past_key_values` 는 보통 post-RoPE를 저장한다. 우리는 pre-RoPE로 저장하기로 했으므로, attention 모듈이 K에 RoPE를 적용하기 직전에 hook을 걸거나, 적용 후 un-rotate해서 저장한다. 이 부분이 가장 까다로움.
2. **Position ID**: 청크가 위치 P에서 시작한다면 position_ids는 [P, P+1, ..., P+S-1]. 시스템 프롬프트는 [0, ..., len-1]. 첫 문서는 [len, ..., len+S-1]. 등등.
3. **Attention mask in fused prefill**: 모든 청크가 concat된 후, 표준 causal mask가 적용된다. 청크 간 mask가 아니다.
4. **Tokenizer 일관성**: 청크 단독 토큰화 결과 != concat 후 토큰화. 보통 토크나이저는 BOS, 공백 처리에서 차이를 만든다. 가장 안전한 방법: 전체 입력을 토큰화한 후 분할.
