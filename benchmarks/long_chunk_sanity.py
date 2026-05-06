"""
Long-chunk sanity (Phase 4 gate) — does ratio=0.15's L2 reduction grow with
chunk length, as Phase 3's report predicted?

Phase 3 measured ~23 % reduction at ratio=0.15 on a 20-token, 2-chunk input
(chunk B = 10 tokens). The report attributed this to the small-input ceiling
(15 % of 10 = 1.5 tokens, capping recoverable error at ~30 %). If the
algorithm is implemented correctly, longer chunk B's should show larger
reductions and converge towards the paper's 50 %+ figures.

Decision rule:
  Pass → at chunk B ∈ {100, 200} tokens, ratio=0.15 reduction ≥ 40 %.
  Fail → reduction stays < 30 % even with longer chunks; algorithm bug
         possible; stop and ask user.

Usage:
  python benchmarks/long_chunk_sanity.py

Output: Markdown-ready table + Insight 2 overlap per length.
"""
from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cacheblend.chunker import Chunk, _hash_text
from cacheblend.fusor import fuse_full_reuse, fuse_selective
from cacheblend.hkvd import select_top_k
from cacheblend.kv_store import KVStore
from cacheblend.model import LayerwiseModel
from cacheblend.precompute import precompute_chunk_kv_from_ids


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
CHUNK_A_TEXT = "Background context: "
CHUNK_B_LENS = [50, 100, 200]
RATIOS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.50]
CHECK_LAYER = 1


def _chunk_from_ids(text: str, ids: torch.Tensor, position: int) -> Chunk:
    return Chunk(
        text=text,
        token_ids=ids.to(torch.long),
        position=position,
        hash=_hash_text(text),
    )


def _build_chunk_b(target_len: int, tokenizer):
    """Build a chunk B whose tokenization is exactly target_len tokens.

    We tokenize a long lorem-ipsum source and slice to the desired length.
    Storing the (text, ids) pair is enough for chunk_texts-equivalent semantics —
    we override the chunk's ids directly so the fused sequence matches.
    """
    base = (
        "The Eiffel Tower, located in Paris, France, was completed in 1889 as the "
        "centerpiece of the World's Fair celebrating the centennial of the French "
        "Revolution. It was the tallest man-made structure in the world for forty-one "
        "years until the Chrysler Building in New York City was finished in 1930. "
        "The tower has three levels for visitors and remains the most-visited paid "
        "monument on Earth, drawing nearly seven million tourists every year. "
    )
    # Repeat enough to surely exceed target_len then slice.
    text = base * 6
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    if int(ids.shape[0]) < target_len:
        raise RuntimeError(
            f"base text too short: got {int(ids.shape[0])}, need {target_len}"
        )
    ids = ids[:target_len]
    decoded = tokenizer.decode(ids)
    return decoded, ids


@torch.no_grad()
def main():
    print(f"# Long-chunk sanity — {MODEL_ID}, FP32 CPU")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, attn_implementation="eager"
    ).to("cpu").eval()
    lw = LayerwiseModel(hf, dtype=torch.float32, device="cpu", kv_form="pre_rope")
    print(f"loaded {time.time() - t0:.1f}s", flush=True)

    a_ids = tokenizer(
        CHUNK_A_TEXT, add_special_tokens=False, return_tensors="pt"
    ).input_ids[0].to(torch.long)
    a_len = int(a_ids.shape[0])
    print(f"chunk A: '{CHUNK_A_TEXT}' → {a_len} tokens", flush=True)

    decision = "pending"
    for b_target in CHUNK_B_LENS:
        b_text, b_ids = _build_chunk_b(b_target, tokenizer)
        b_len = int(b_ids.shape[0])
        chunks = [
            _chunk_from_ids(CHUNK_A_TEXT, a_ids, position=0),
            _chunk_from_ids(b_text, b_ids, position=a_len),
        ]
        S_total = a_len + b_len
        full_ids = torch.cat([a_ids, b_ids]).unsqueeze(0)

        store = KVStore()
        for c in chunks:
            store.put(c.hash, precompute_chunk_kv_from_ids(lw, c.token_ids))

        t = time.time()
        recompute = lw.forward_layerwise(full_ids)
        t_recomp = time.time() - t
        t = time.time()
        reuse = fuse_full_reuse(lw, chunks, store)
        t_reuse = time.time() - t
        l2_reuse = (reuse - recompute).norm().item()

        print(
            f"\n## chunk B = {b_len} tokens (S_total = {S_total})  "
            f"[recompute {t_recomp:.1f}s, reuse {t_reuse:.1f}s, L2(reuse) = {l2_reuse:.3e}]",
            flush=True,
        )
        print("| ratio | L2 vs recompute | L2 / L2(reuse) | reduction % |")
        print("|---|---|---|---|")
        results = {}
        for r in RATIOS:
            t = time.time()
            sel = fuse_selective(lw, chunks, store, recompute_ratio=r, check_layer=CHECK_LAYER)
            l2 = (sel - recompute).norm().item()
            ratio_to_reuse = l2 / l2_reuse if l2_reuse > 0 else float("nan")
            reduction = (1.0 - ratio_to_reuse) * 100.0
            print(
                f"| {r:.2f} | {l2:.3e} | {ratio_to_reuse:.3f} | {reduction:+.1f}% | "
                f"({time.time() - t:.1f}s)"
            )
            results[r] = (l2, ratio_to_reuse, reduction)

        # Phase 4 gate decision based on ratio=0.15.
        red_15 = results[0.15][2]
        print(f"\n  → ratio=0.15 reduction at chunk B = {b_len}: {red_15:.1f}%", flush=True)

        # Insight 2 overlap at this chunk length.
        from cacheblend.fusor import (
            _gather_cached_kv_per_layer,
            _make_synth_hook,
        )
        from cacheblend.rope import _rotate_half
        from transformers.cache_utils import DynamicCache

        def deviation_at_layer(target_layer: int) -> torch.Tensor:
            B = 1
            hidden = lw.embed_tokens(full_ids)
            pos_ids = torch.arange(S_total).unsqueeze(0).expand(B, -1)
            cache_pos = torch.arange(S_total)
            pe = lw.compute_position_embeddings(hidden, pos_ids)
            cache = DynamicCache(config=lw.config)
            mask = lw.build_causal_mask(hidden, pos_ids, cache_pos, cache)
            cK, cV = _gather_cached_kv_per_layer(chunks, store, lw.num_layers, "cpu", lw.dtype)
            keep_empty = torch.zeros(S_total, dtype=torch.bool)
            for li in range(target_layer):
                kh = lw._inner.layers[li].self_attn.k_proj.register_forward_hook(
                    _make_synth_hook(cK[li], keep_empty)
                )
                vh = lw._inner.layers[li].self_attn.v_proj.register_forward_hook(
                    _make_synth_hook(cV[li], keep_empty)
                )
                try:
                    out = lw.prefill_layer(
                        layer_idx=li, hidden_states=hidden, position_ids=pos_ids,
                        position_embeddings=pe, attention_mask=mask,
                        past_key_values=cache, cache_position=cache_pos,
                    )
                    hidden = out.hidden
                finally:
                    kh.remove(); vh.remove()
            layer = lw._inner.layers[target_layer]
            normed = layer.input_layernorm(hidden)
            fresh = layer.self_attn.k_proj(normed)
            B_, S_, _ = fresh.shape
            fresh_pre = fresh.view(B_, S_, lw.num_kv_heads, lw.head_dim).transpose(1, 2).contiguous()
            cos, sin = pe
            cos_b, sin_b = cos.unsqueeze(1), sin.unsqueeze(1)
            fresh_post = (fresh_pre * cos_b) + (_rotate_half(fresh_pre) * sin_b)
            cached_pre = cK[target_layer]
            cached_post = (cached_pre * cos_b) + (_rotate_half(cached_pre) * sin_b)
            diff = fresh_post.float() - cached_post.float()
            return (diff * diff).sum(dim=(1, 3))[0]

        k15 = max(1, int(round(S_total * 0.15)))
        s_layer1 = set(select_top_k(deviation_at_layer(1), k15).tolist())
        s_last = set(select_top_k(deviation_at_layer(lw.num_layers - 1), k15).tolist())
        overlap = len(s_layer1 & s_last) / k15
        print(
            f"  Insight 2: layer 1 vs layer {lw.num_layers - 1}, k={k15}, "
            f"overlap = {len(s_layer1 & s_last)}/{k15} = {overlap:.3f}"
        )

    print(f"\ntotal: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
