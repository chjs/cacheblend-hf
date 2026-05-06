"""Phase 1 layerwise bit-exact regression for the Phase-5 model on GPU.

Run on vast.ai:
  PYTHONPATH=. python scripts/phase5_layerwise_regression.py \
      --model Qwen/Qwen2.5-7B-Instruct --dtype float16 --device cuda

Tolerance: FP16 → 1e-3 (consistent with Phase 1 spec). FP32 → 1e-5.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cacheblend.model import LayerwiseModel


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--text", default="The Eiffel Tower stands in Paris.")
    p.add_argument("--output", type=Path, default=Path("benchmarks/results/phase1_regression.json"))
    args = p.parse_args()

    print(f"[load] {args.model} ({args.dtype}, {args.device})", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    torch_dtype = getattr(torch, args.dtype)
    hf = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, attn_implementation="eager"
    ).to(args.device).eval()
    lw = LayerwiseModel(hf, dtype=torch_dtype, device=args.device, kv_form="pre_rope")
    print(f"  loaded {time.time() - t0:.1f}s", flush=True)

    input_ids = tokenizer(args.text, return_tensors="pt").input_ids.to(args.device)
    print(f"[forward] S = {int(input_ids.shape[1])}", flush=True)

    with torch.no_grad():
        std = hf(input_ids=input_ids, use_cache=False).logits
        if args.device == "cuda":
            torch.cuda.synchronize()
        lw_logits = lw.forward_layerwise(input_ids)
        if args.device == "cuda":
            torch.cuda.synchronize()

    diff = (std - lw_logits).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    tol = 1e-3 if args.dtype == "float16" else (1e-5 if args.dtype == "float32" else 5e-3)
    ok = max_diff < tol
    print(
        f"[result] max_diff = {max_diff:.3e}  mean_diff = {mean_diff:.3e}  "
        f"tol = {tol:.0e}  {'PASS' if ok else 'FAIL'}",
        flush=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "model": args.model,
        "dtype": args.dtype,
        "device": args.device,
        "S": int(input_ids.shape[1]),
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "tolerance": tol,
        "pass": ok,
    }, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
