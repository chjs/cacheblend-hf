# vast.ai Playbook

> When and how to use vast.ai for this project. Read once; come back when Phase 4 starts.

## When to use vast.ai vs local Mac

| Phase | Where | Why |
|---|---|---|
| Phase 0 (Setup) | Local Mac | No GPU needed |
| Phase 1 (Layerwise forward) | Local Mac, Qwen2.5-1.5B FP32 on CPU | Bit-exact verification works on CPU |
| Phase 2 (KV storage) | Local Mac | Synthetic, small model |
| Phase 3 (Selective recompute) | Local Mac | Same |
| Phase 4 (Pipelining) | **vast.ai** | TTFT measurements need real GPU, real disk |
| Phase 5 (Evaluation) | **vast.ai** | Mistral-7B + datasets |

Optional: Even Phase 1~3 can be cross-verified on vast.ai once before signing off, to catch CPU-vs-CUDA numerical surprises early.

## Instance recommendation

For Phase 5 (Mistral-7B FP16):

```
vastai search offers \
  'reliability > 0.99 num_gpus=1 \
   gpu_ram >= 24 \
   cuda_max_good >= 12.1 \
   inet_down >= 100 \
   dlperf_per_dphtotal >= 1500 \
   verified=True'
```

- `gpu_ram >= 24`: Mistral-7B FP16 + ~4K context KV cache fits comfortably; A40 (48GB) or RTX 6000 Ada is ideal but more expensive than RTX 4090 (24GB)
- `cuda_max_good >= 12.1`: matches PyTorch 2.4 CUDA 12.x
- `inet_down >= 100`: faster HF model download
- `verified=True`: stable hosts only

For initial sanity (Phase 4 light tests), an RTX 4090 is sufficient and cheap.

## Persistent model cache pattern

The whole point of paying for storage is so you don't re-download Mistral-7B (~14GB) every time.

1. **Create instance** with enough disk:

   ```bash
   bash scripts/vast.sh up <OFFER_ID>
   ```

   `vast.sh up` allocates `--disk 80` (covers Mistral-7B + Qwen + caches + KV files), uses the `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime` image, and runs an `--onstart-cmd` that sets `HF_HOME=/workspace/hf_cache`.

2. **First run only**: download models. They land in `/workspace/hf_cache`, which is persistent across stop/start (but **not** destroy).

3. **When done for the day**:

   ```bash
   bash scripts/vast.sh stop
   ```

   GPU billing stops. Disk billing continues at a lower rate.

4. **Resume**:

   ```bash
   bash scripts/vast.sh start
   bash scripts/vast.sh ssh
   # cache is still there, no re-download
   ```

5. **Permanently done**:

   ```bash
   bash scripts/vast.sh destroy
   ```

## Code sync pattern

Don't `git pull` on the instance every time — use rsync from your Mac:

```bash
# Mac → instance (push code changes)
bash scripts/vast.sh push

# instance → Mac (pull results / reports)
bash scripts/vast.sh pull
```

Under the hood: `rsync -avz --exclude` with sensible excludes (`.venv`, `__pycache__`, `external/LMCache`, `*.pt`, etc.).

## Cost discipline

- `vastai show instances` in a habit before going to bed.
- Set a daily budget alarm in vast.ai dashboard.
- For long-running runs, prefer `tmux` on the instance so an SSH disconnect doesn't kill your job.

## Common gotchas

1. **Env vars don't appear in SSH session**: vast.ai sets env via `/etc/environment` only if your `--onstart-cmd` writes them there. Our `vast.sh up` does this.
2. **Storage charges during stop**: ~$0.10/GB-month on most hosts. 80GB ≈ $8/month. Destroy if you won't use it for weeks.
3. **First-time HF model download is slow**: 14GB Mistral over typical 100Mbps takes ~20 minutes. Worth it once.
4. **`--ssh --direct`**: direct SSH is faster than proxy SSH but requires the host to support it. Most verified hosts do.
5. **Wrong CUDA**: If you pin torch 2.4 + cu121, pick a host with `cuda_max_good >= 12.1`. Mismatch causes silent slowness or hard crashes.

## Reference: useful raw commands

```bash
# Search (price-sorted)
vastai search offers '...' -o 'dphtotal'

# Show what's running
vastai show instances

# SSH command for an instance
vastai ssh-url <INSTANCE_ID>

# Logs (debugging onstart-cmd)
vastai logs <INSTANCE_ID>

# Resize disk (only when stopped)
vastai change instance <INSTANCE_ID> --disk 120
```
