"""Phase 5 — Insight 2 (RAG): HKVD overlap + Spearman across layers.

Run on vast.ai (or anywhere with the model on disk):
  PYTHONPATH=. python scripts/phase5_insight2_rag.py \
      --model Qwen/Qwen2.5-7B-Instruct --dtype float16 --device cuda \
      --limit 5 --chunk-size 512

For each Musique example, compute per-token KV deviation at layer 1 and at
the last layer (layer L-1) using the full-reuse pre-pass machinery, then
report:
  - top-k overlap (k = round(0.15 × S))
  - Spearman rank correlation of the two deviation vectors

Outputs JSON with per-example numbers + an aggregate summary.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from benchmarks.datasets import musique
from benchmarks.rag import build_rag_input
from benchmarks.run_benchmark import _chunks_from_rag, _seed_store
from cacheblend.fusor import _gather_cached_kv_per_layer, _make_synth_hook
from cacheblend.hkvd import select_top_k
from cacheblend.model import LayerwiseModel
from cacheblend.rope import _rotate_half


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """1 - 6 Σ d² / (n (n² − 1)) with average ranks for ties."""
    n = a.numel()
    if n < 2:
        return float("nan")
    ar = torch.argsort(torch.argsort(a)).float()
    br = torch.argsort(torch.argsort(b)).float()
    d = ar - br
    return 1 - 6 * (d * d).sum().item() / (n * (n * n - 1))


@torch.no_grad()
def deviation_at_layer(
    lw: LayerwiseModel,
    chunks,
    store,
    target_layer: int,
) -> torch.Tensor:
    device = lw.device
    dtype = lw.dtype
    full_ids = torch.cat([c.token_ids for c in chunks]).unsqueeze(0).to(device)
    B, S = full_ids.shape

    hidden = lw.embed_tokens(full_ids)
    pos_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    cache_pos = torch.arange(S, device=device)
    pe = lw.compute_position_embeddings(hidden, pos_ids)
    layer_cache = DynamicCache(config=lw.config)
    mask = lw.build_causal_mask(hidden, pos_ids, cache_pos, layer_cache)
    cached_K, cached_V = _gather_cached_kv_per_layer(
        chunks, store, lw.num_layers, device, dtype
    )

    keep_empty = torch.zeros(S, dtype=torch.bool, device=device)
    for li in range(target_layer):
        layer = lw._inner.layers[li]
        kh = layer.self_attn.k_proj.register_forward_hook(
            _make_synth_hook(cached_K[li], keep_empty)
        )
        vh = layer.self_attn.v_proj.register_forward_hook(
            _make_synth_hook(cached_V[li], keep_empty)
        )
        try:
            out = lw.prefill_layer(
                layer_idx=li, hidden_states=hidden, position_ids=pos_ids,
                position_embeddings=pe, attention_mask=mask,
                past_key_values=layer_cache, cache_position=cache_pos,
            )
            hidden = out.hidden
        finally:
            kh.remove()
            vh.remove()

    layer = lw._inner.layers[target_layer]
    normed = layer.input_layernorm(hidden)
    fresh = layer.self_attn.k_proj(normed)
    fresh_pre = (
        fresh.view(1, S, lw.num_kv_heads, lw.head_dim).transpose(1, 2).contiguous()
    )
    cos, sin = pe
    cos_b, sin_b = cos.unsqueeze(1), sin.unsqueeze(1)
    fresh_post = (fresh_pre * cos_b) + (_rotate_half(fresh_pre) * sin_b)
    cached_pre = cached_K[target_layer]
    cached_post = (cached_pre * cos_b) + (_rotate_half(cached_pre) * sin_b)
    diff = fresh_post.float() - cached_post.float()
    return (diff * diff).sum(dim=(1, 3))[0].cpu()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--ratio", type=float, default=0.15)
    p.add_argument("--output", type=Path, default=Path("benchmarks/results/insight2_rag.json"))
    args = p.parse_args()

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    torch_dtype = getattr(torch, args.dtype)
    hf = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, attn_implementation="eager"
    ).to(args.device).eval()
    lw = LayerwiseModel(hf, dtype=torch_dtype, device=args.device, kv_form="pre_rope")
    print(f"loaded {time.time() - t0:.1f}s", flush=True)

    items = musique.load(limit=args.limit)
    print(f"[insight2] {len(items)} examples", flush=True)

    rows = []
    last = lw.num_layers - 1
    for idx, it in enumerate(items):
        rag = build_rag_input(it, tokenizer, chunk_size=args.chunk_size)
        chunks = _chunks_from_rag(rag, tokenizer)
        S = sum(int(c.token_ids.shape[0]) for c in chunks)
        store = _seed_store(lw, chunks)
        d1 = deviation_at_layer(lw, chunks, store, target_layer=1)
        dL = deviation_at_layer(lw, chunks, store, target_layer=last)
        k = max(1, int(round(S * args.ratio)))
        s1 = set(select_top_k(d1, k).tolist())
        sL = set(select_top_k(dL, k).tolist())
        overlap = len(s1 & sL) / k
        rho = _spearman(d1, dL)
        rows.append({"id": it["id"], "S": S, "k": k, "overlap": overlap, "spearman": rho})
        print(
            f"[{idx + 1}/{len(items)}] S={S}  k={k}  overlap={overlap:.3f}  spearman={rho:+.3f}",
            flush=True,
        )

    overlaps = [r["overlap"] for r in rows]
    rhos = [r["spearman"] for r in rows]
    summary = {
        "model": args.model,
        "dtype": args.dtype,
        "n": len(rows),
        "ratio": args.ratio,
        "overlap_mean": sum(overlaps) / len(overlaps) if overlaps else 0.0,
        "overlap_min": min(overlaps) if overlaps else 0.0,
        "overlap_max": max(overlaps) if overlaps else 0.0,
        "spearman_mean": sum(rhos) / len(rhos) if rhos else 0.0,
        "spearman_min": min(rhos) if rhos else 0.0,
        "spearman_max": max(rhos) if rhos else 0.0,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
