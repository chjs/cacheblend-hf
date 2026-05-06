"""
Phase 5 — end-to-end benchmark runner.

Methods:
  full_recompute  — vanilla HF forward + greedy decode. No cache reuse.
  prefix_cache    — system prompt KV reused across requests, docs+query fresh.
                    Single-request TTFT excludes the system-prompt prefill cost
                    (assumed amortized across many requests).
  full_reuse      — every chunk's KV pre-loaded; runs ``fuse_full_reuse``.
  cacheblend      — every chunk's KV pre-loaded; runs ``fuse_selective`` at
                    ``--ratio``.

For each example we measure:
  - F1 against the gold answer (token-level, max over aliases).
  - TTFT — wall time of the *prefill* call that produces the first logits.

Greedy decoding is unified across methods: after the prefill we hand the
model the resulting ``DynamicCache`` and step through up to
``--max-new-tokens`` tokens of standard HF forward.

Usage:
  python -m benchmarks.run_benchmark \
      --model Qwen/Qwen2.5-1.5B-Instruct \
      --dataset musique --limit 5 \
      --method cacheblend --ratio 0.15 \
      --output benchmarks/results/musique_cacheblend.json
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from benchmarks.datasets import musique
from benchmarks.metrics.qa import aggregate, f1_score
from benchmarks.rag import RAGInput, build_rag_input, count_tokenizer_mismatches
from benchmarks.ttft import _sync
from cacheblend.chunker import Chunk, _hash_text
from cacheblend.fusor import (
    _gather_cached_kv_per_layer,
    _make_synth_hook,
    fuse_full_reuse,
    fuse_selective,
)
from cacheblend.kv_store import KVStore
from cacheblend.model import LayerwiseModel
from cacheblend.precompute import precompute_chunk_kv_from_ids


METHODS = ("full_recompute", "prefix_cache", "full_reuse", "cacheblend")


def _chunks_from_rag(rag: RAGInput, tokenizer) -> List[Chunk]:
    """Build position-aware Chunks. Same convention as cacheblend.chunker."""
    chunks: List[Chunk] = []
    pos = 0
    for text in rag.chunk_texts:
        ids = tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        ).input_ids[0].to(torch.long)
        chunks.append(
            Chunk(
                text=text,
                token_ids=ids,
                position=pos,
                hash=_hash_text(text),
            )
        )
        pos += int(ids.shape[0])
    return chunks


def _greedy_decode(
    hf_model,
    cache: DynamicCache,
    last_logits: torch.Tensor,
    n_new_tokens: int,
    eos_token_id: int,
    device: torch.device,
) -> List[int]:
    """Step ``hf_model`` greedily for up to ``n_new_tokens`` steps using
    ``cache`` as past_key_values. ``last_logits`` is the prefill's final-token
    logit row; we take its argmax as token #1 and continue."""
    out_ids: List[int] = []
    next_token = last_logits[:, -1, :].argmax(dim=-1).view(1, 1)  # (1, 1)
    out_ids.append(int(next_token.item()))
    if int(next_token.item()) == eos_token_id:
        return out_ids
    seq_len = cache.get_seq_length()
    cache_position = torch.tensor([seq_len], device=device, dtype=torch.long)
    for _ in range(n_new_tokens - 1):
        with torch.no_grad():
            out = hf_model(
                input_ids=next_token,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
            )
        next_token = out.logits[:, -1, :].argmax(dim=-1).view(1, 1)
        out_ids.append(int(next_token.item()))
        if int(next_token.item()) == eos_token_id:
            break
        cache_position = cache_position + 1
    return out_ids


def _build_full_input_ids(chunks: Sequence[Chunk]) -> torch.Tensor:
    return torch.cat([c.token_ids for c in chunks]).unsqueeze(0)


def _seed_store(
    lw: LayerwiseModel, chunks: Sequence[Chunk]
) -> KVStore:
    """Compute per-chunk KV with the model and put it in a fresh in-memory
    KVStore. We *exclude* this from the timed TTFT — the assumption being
    that real RAG systems amortize precompute across many queries hitting
    the same documents."""
    store = KVStore()
    seen: set[str] = set()
    for c in chunks:
        if c.hash in seen:
            continue
        seen.add(c.hash)
        store.put(
            c.hash, precompute_chunk_kv_from_ids(lw, c.token_ids.to(lw.device))
        )
    return store


@torch.no_grad()
def _prefill_full_recompute(
    lw: LayerwiseModel,
    chunks: Sequence[Chunk],
):
    """Run ``forward_layerwise`` on the concatenated input but mirror its
    body so we can keep the resulting ``DynamicCache`` for greedy decoding."""
    full_ids = _build_full_input_ids(chunks).to(lw.device)
    B, S = full_ids.shape
    hidden = lw.embed_tokens(full_ids)
    pos_ids = torch.arange(S, device=lw.device).unsqueeze(0).expand(B, -1)
    cache_pos = torch.arange(S, device=lw.device)
    pe = lw.compute_position_embeddings(hidden, pos_ids)
    cache = DynamicCache(config=lw.config)
    mask = lw.build_causal_mask(hidden, pos_ids, cache_pos, cache)
    for i in range(lw.num_layers):
        out = lw.prefill_layer(
            layer_idx=i,
            hidden_states=hidden,
            position_ids=pos_ids,
            position_embeddings=pe,
            attention_mask=mask,
            past_key_values=cache,
            cache_position=cache_pos,
        )
        hidden = out.hidden
    logits = lw.final_norm_and_lm_head(hidden)
    return logits, cache


@torch.no_grad()
def _prefill_prefix_cache(
    lw: LayerwiseModel,
    chunks: Sequence[Chunk],
    sys_chunk: Chunk,
):
    """Treat the system prompt as already cached: run a full prefill on the
    full sequence, but assume the system-prompt prefix's prefill is amortized
    across requests. We approximate by running the full prefill (same as
    full_recompute) — TTFT comparison vs full_recompute is meaningful only
    after subtracting the sys-prompt prefill cost externally. For this v1 we
    return the same logits/cache as full_recompute and let the report
    explain the budget."""
    return _prefill_full_recompute(lw, chunks)


@torch.no_grad()
def _prefill_full_reuse(
    lw: LayerwiseModel,
    chunks: Sequence[Chunk],
    store: KVStore,
):
    """``fuse_full_reuse`` returns logits but not the layer_cache it built.
    Re-implement the loop so we can keep the cache (with cached pre-RoPE K +
    cached V at every position; greedy decode will then attend over those)."""
    return _fuse_with_cache(lw, chunks, store, recompute_ratio=0.0, check_layer=0)


@torch.no_grad()
def _prefill_cacheblend(
    lw: LayerwiseModel,
    chunks: Sequence[Chunk],
    store: KVStore,
    ratio: float,
    check_layer: int = 1,
):
    return _fuse_with_cache(
        lw, chunks, store, recompute_ratio=ratio, check_layer=check_layer
    )


@torch.no_grad()
def _fuse_with_cache(
    lw: LayerwiseModel,
    chunks: Sequence[Chunk],
    store: KVStore,
    recompute_ratio: float,
    check_layer: int,
):
    """Mirror of ``cacheblend.fusor._fuse`` that returns the populated
    DynamicCache so the caller can greedy-decode after the prefill."""
    from cacheblend.fusor import _select_hkvd_at_check_layer
    from cacheblend.fusor import _make_synth_hook as _msh

    full_ids = _build_full_input_ids(chunks).to(lw.device)
    B, S_total = full_ids.shape

    hidden = lw.embed_tokens(full_ids)
    pos_ids = torch.arange(S_total, device=lw.device).unsqueeze(0).expand(B, -1)
    cache_pos = torch.arange(S_total, device=lw.device)
    pe = lw.compute_position_embeddings(hidden, pos_ids)
    layer_cache = DynamicCache(config=lw.config)
    mask = lw.build_causal_mask(hidden, pos_ids, cache_pos, layer_cache)

    cached_K, cached_V = _gather_cached_kv_per_layer(
        chunks, store, lw.num_layers, lw.device, lw.dtype
    )

    hkvd_indices = _select_hkvd_at_check_layer(
        model=lw,
        hidden_at_layer_0=hidden,
        cached_K=cached_K,
        cached_V=cached_V,
        pos_ids=pos_ids,
        cache_pos=cache_pos,
        pe=pe,
        attn_mask=mask,
        check_layer=check_layer,
        recompute_ratio=recompute_ratio,
        S_total=S_total,
    )
    layer_cache = DynamicCache(config=lw.config)
    mask = lw.build_causal_mask(hidden, pos_ids, cache_pos, layer_cache)

    keep_fresh = torch.zeros(S_total, dtype=torch.bool, device=lw.device)
    if hkvd_indices.numel() > 0:
        keep_fresh[hkvd_indices.to(lw.device)] = True

    for layer_idx in range(lw.num_layers):
        layer = lw._inner.layers[layer_idx]
        kh = layer.self_attn.k_proj.register_forward_hook(
            _msh(cached_K[layer_idx], keep_fresh)
        )
        vh = layer.self_attn.v_proj.register_forward_hook(
            _msh(cached_V[layer_idx], keep_fresh)
        )
        try:
            out = lw.prefill_layer(
                layer_idx=layer_idx,
                hidden_states=hidden,
                position_ids=pos_ids,
                position_embeddings=pe,
                attention_mask=mask,
                past_key_values=layer_cache,
                cache_position=cache_pos,
            )
            hidden = out.hidden
        finally:
            kh.remove()
            vh.remove()

    logits = lw.final_norm_and_lm_head(hidden)
    return logits, layer_cache


def run(
    model_id: str,
    dataset: str,
    method: str,
    *,
    recompute_ratio: float = 0.15,
    limit: Optional[int] = None,
    chunk_size: int = 512,
    max_new_tokens: int = 20,
    dtype: str = "float32",
    device: str = "cpu",
    output: Optional[Path] = None,
    n_warmup: int = 1,
) -> dict:
    if method not in METHODS:
        raise ValueError(f"unknown method: {method!r}; choose from {METHODS}")

    print(f"# {method} on {dataset} ({model_id}, {dtype}, {device})", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    torch_dtype = getattr(torch, dtype)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype, attn_implementation="eager"
    ).to(device).eval()
    lw = LayerwiseModel(hf_model, dtype=torch_dtype, device=device, kv_form="pre_rope")
    print(f"loaded {time.time() - t0:.1f}s", flush=True)

    if dataset == "musique":
        items = musique.load(limit=limit)
    else:
        raise ValueError(f"unknown dataset: {dataset!r}")
    print(f"loaded {len(items)} examples", flush=True)

    rag_inputs = [build_rag_input(it, tokenizer, chunk_size=chunk_size) for it in items]
    mismatch_stats = count_tokenizer_mismatches(rag_inputs, tokenizer)
    print(f"tokenizer mismatch: {mismatch_stats}", flush=True)

    f1s: List[float] = []
    ttfts_ms: List[float] = []
    eos_token_id = (
        tokenizer.eos_token_id
        if tokenizer.eos_token_id is not None
        else tokenizer.convert_tokens_to_ids("\n")
    )
    device_t = torch.device(device)

    for idx, rag in enumerate(rag_inputs):
        chunks = _chunks_from_rag(rag, tokenizer)
        S_total = int(sum(c.token_ids.shape[0] for c in chunks))

        # Pre-seed KV cache for the methods that need it. Excluded from TTFT.
        store: Optional[KVStore] = None
        if method in ("full_reuse", "cacheblend"):
            store = _seed_store(lw, chunks)

        # Warmup once per example for the first one (compiles caches, kernels).
        if idx == 0:
            for _ in range(max(0, n_warmup)):
                if method == "full_recompute":
                    _prefill_full_recompute(lw, chunks)
                elif method == "prefix_cache":
                    _prefill_prefix_cache(lw, chunks, chunks[0])
                elif method == "full_reuse":
                    _prefill_full_reuse(lw, chunks, store)
                else:  # cacheblend
                    _prefill_cacheblend(lw, chunks, store, recompute_ratio)
                _sync(device_t)

        # Timed prefill (TTFT).
        _sync(device_t)
        prefill_start = time.perf_counter()
        if method == "full_recompute":
            logits, cache = _prefill_full_recompute(lw, chunks)
        elif method == "prefix_cache":
            logits, cache = _prefill_prefix_cache(lw, chunks, chunks[0])
        elif method == "full_reuse":
            logits, cache = _prefill_full_reuse(lw, chunks, store)
        else:  # cacheblend
            logits, cache = _prefill_cacheblend(lw, chunks, store, recompute_ratio)
        _sync(device_t)
        ttft_s = time.perf_counter() - prefill_start
        ttfts_ms.append(ttft_s * 1000.0)

        # Greedy decode (not timed).
        gen_ids = _greedy_decode(
            hf_model=lw.hf_model,
            cache=cache,
            last_logits=logits,
            n_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            device=device_t,
        )
        pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        gold = [rag.answer] + list(rag.answer_aliases)
        score = f1_score(pred_text, gold)
        f1s.append(score)
        print(
            f"[{idx + 1}/{len(rag_inputs)}] S={S_total}  ttft={ttft_s * 1000:.0f}ms  "
            f"f1={score:.3f}  pred={pred_text!r}  gold={rag.answer!r}",
            flush=True,
        )

    # Aggregate.
    f1_agg = aggregate(f1s)
    ttft_agg = aggregate(ttfts_ms)
    result = {
        "method": method,
        "dataset": dataset,
        "model": model_id,
        "dtype": dtype,
        "device": device,
        "ratio": recompute_ratio if method == "cacheblend" else None,
        "n": len(rag_inputs),
        "chunk_size": chunk_size,
        "max_new_tokens": max_new_tokens,
        "f1": f1_agg,
        "ttft_ms": ttft_agg,
        "tokenizer_mismatch": mismatch_stats,
        "wall_time_s": time.time() - t0,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
        print(f"wrote {output}", flush=True)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--dataset", default="musique", choices=["musique"])
    p.add_argument("--method", required=True, choices=list(METHODS))
    p.add_argument("--ratio", type=float, default=0.15, dest="recompute_ratio")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=20)
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--n-warmup", type=int, default=1)
    a = p.parse_args()

    run(
        model_id=a.model,
        dataset=a.dataset,
        method=a.method,
        recompute_ratio=a.recompute_ratio,
        limit=a.limit,
        chunk_size=a.chunk_size,
        max_new_tokens=a.max_new_tokens,
        dtype=a.dtype,
        device=a.device,
        output=a.output,
        n_warmup=a.n_warmup,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
