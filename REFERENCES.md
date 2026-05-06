# References

## Primary

- **CacheBlend paper** (this is the spec we are implementing):
  Yao et al., "CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion", EuroSys 2025.
  arXiv: https://arxiv.org/abs/2405.16444
  ACM: https://doi.org/10.1145/3689031.3696098

- **LMCache** (the official implementation. We study, but do not copy):
  https://github.com/LMCache/LMCache
  - The CacheBlend logic lives roughly in `lmcache/blend/` and the engine integration. `tasks/phase-0-analysis.md` walks through which parts to read.

## Foundational

- **Rotary Positional Embedding (RoPE)** — Su et al., "RoFormer", Neurocomputing 2024.
  Used by the Fusor when re-positioning a cached chunk's K vectors.

- **PromptCache** — Gim et al., 2023. The "full KV reuse" baseline whose limitation CacheBlend addresses.
  arXiv: https://arxiv.org/abs/2311.04934

- **vLLM PagedAttention** — Kwon et al., SOSP 2023. The serving engine LMCache integrates with. We do **not** integrate; we stay on vanilla HF Transformers.
  arXiv: https://arxiv.org/abs/2309.06180

## Datasets

- **Musique** — Trivedi et al., "MuSiQue: Multihop Questions via Single-hop Question Composition".
  https://github.com/StonyBrookNLP/musique
- **2WikiMultihopQA** — Ho et al., 2020.
  https://github.com/Alab-NII/2wikimultihop
- **SAMSum** — Gliwa et al., 2019.
  https://huggingface.co/datasets/samsum
- **MultiNews** — Fabbri et al., 2019.
  https://huggingface.co/datasets/multi_news

## Useful HF Transformers internals

- Decoder-only model layer source (Mistral example):
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py
- Cache classes (`DynamicCache`, `StaticCache`):
  https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py

## Concept cheat sheet

- **Prefix caching** — only cache prefix; works perfectly when reused chunk is at the start.
- **Full KV reuse** — concatenate cached KV regardless of position; loses cross-attention.
- **Selective KV recompute (CacheBlend)** — full reuse + recompute KV of HKVD tokens.
- **HKVD** — *High KV Deviation*. Tokens whose KV (in the cached, position-shifted form) most differs from the would-be recomputed KV.
- **Gradual filtering** — Layer 1 picks r1% candidates, layer 2 narrows to r2% (r2<r1) among those, ...
- **Cross-attention (in this paper's usage)** — attention between tokens in chunk A and tokens in chunk B (≠ encoder-decoder cross-attention).
- **Forward attention matrix** — attention from the *last* (query-side) tokens onto all preceding tokens. This is what directly affects the next generated token.
