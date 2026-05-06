"""Phase 5 local plumbing smoke: 1 Musique sample × first 3 paragraphs ×
all 4 methods × tiny chunk_size on Mac CPU FP32 with Qwen2.5-1.5B.

Goal: detect dimension mismatches, missing functions, broken decode loop.
Not a quality measurement.
"""
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks.datasets import musique
from benchmarks.metrics.qa import f1_score
from benchmarks.rag import build_rag_input
from benchmarks.run_benchmark import (
    _chunks_from_rag,
    _greedy_decode,
    _prefill_cacheblend,
    _prefill_full_recompute,
    _prefill_full_reuse,
    _prefill_prefix_cache,
    _seed_store,
)
from cacheblend.model import LayerwiseModel


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cpu"
DTYPE = torch.float32
CHUNK_SIZE = 96            # tiny so the test fits in human time
N_DOCS = 3                 # truncate to top-3 paragraphs
MAX_NEW_TOKENS = 8


def _truncate_docs(item: dict, n: int) -> dict:
    out = dict(item)
    out["documents"] = item["documents"][:n]
    return out


def main() -> int:
    print("[load model]", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, attn_implementation="eager"
    ).to(DEVICE).eval()
    lw = LayerwiseModel(hf, dtype=DTYPE, device=DEVICE, kv_form="pre_rope")
    print(f"  loaded {time.time() - t0:.1f}s", flush=True)

    print(f"[load 1 musique example]", flush=True)
    items = musique.load(limit=1)
    item = _truncate_docs(items[0], N_DOCS)
    print(f"  question: {item['query']!r}", flush=True)
    print(f"  answer:   {item['answer']!r}", flush=True)

    rag = build_rag_input(item, tokenizer, chunk_size=CHUNK_SIZE)
    chunks = _chunks_from_rag(rag, tokenizer)
    s_total = sum(int(c.token_ids.shape[0]) for c in chunks)
    print(f"  {len(chunks)} chunks, total {s_total} tokens", flush=True)

    eos = tokenizer.eos_token_id

    results = {}
    for method in ["full_recompute", "prefix_cache", "full_reuse", "cacheblend"]:
        print(f"\n[{method}]", flush=True)
        t = time.time()
        if method == "full_recompute":
            logits, cache = _prefill_full_recompute(lw, chunks)
        elif method == "prefix_cache":
            logits, cache = _prefill_prefix_cache(lw, chunks, chunks[0])
        elif method == "full_reuse":
            store = _seed_store(lw, chunks)
            logits, cache = _prefill_full_reuse(lw, chunks, store)
        else:
            store = _seed_store(lw, chunks)
            logits, cache = _prefill_cacheblend(lw, chunks, store, ratio=0.15)
        prefill_s = time.time() - t
        gen_ids = _greedy_decode(
            hf_model=lw.hf_model,
            cache=cache,
            last_logits=logits,
            n_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=eos,
            device=torch.device(DEVICE),
        )
        pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        gold = [item["answer"]] + list(item["answer_aliases"])
        f1 = f1_score(pred, gold)
        print(
            f"  prefill {prefill_s:.1f}s  pred={pred!r}  f1={f1:.3f}",
            flush=True,
        )
        results[method] = {"prefill_s": prefill_s, "pred": pred, "f1": f1}

    print(f"\n[summary] total {time.time() - t0:.1f}s")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
