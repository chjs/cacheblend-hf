# cacheblend-hf

> A minimal implementation of **CacheBlend** ([Yao et al., EuroSys 2025](https://arxiv.org/abs/2405.16444)) on top of HuggingFace Transformers.

CacheBlend speeds up prefill in retrieval-augmented LLM workloads by reusing pre-computed KV caches of multiple text chunks (regardless of position) and selectively recomputing the KV of a small fraction (~15%) of high-deviation tokens to recover cross-attention.

This repo is a **work-driven harness**: each phase is defined by clear acceptance criteria, and progress is reported via email.

## Quick start

```bash
# 1. Clone & install
git clone <this-repo>
cd cacheblend-hf
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env: set HF_TOKEN, GMAIL_APP_PASSWORD, REPORT_EMAIL_TO

# 3. Verify email pipeline
python scripts/send_report.py --phase 0 --dry-run

# 4. Read the goal and roadmap
cat GOAL.md
cat PHASES.md
```

## Repository layout

```
cacheblend-hf/
├── GOAL.md              # ⭐ The unchanging objective. Read first.
├── CLAUDE.md            # Operating rules for the Claude Code agent.
├── PHASES.md            # 6-phase roadmap.
├── ARCHITECTURE.md      # Target architecture diagram & module boundaries.
├── REFERENCES.md        # Paper, LMCache, related work links.
│
├── tasks/               # Per-phase work definitions (the actual instructions).
│   ├── phase-0-analysis.md
│   ├── phase-1-layerwise-forward.md
│   ├── phase-2-kv-storage.md
│   ├── phase-3-selective-recompute.md
│   ├── phase-4-pipelining.md
│   └── phase-5-evaluation.md
│
├── docs/                # Living documentation.
│   ├── paper-summary.md
│   ├── lmcache-analysis.md
│   └── design-decisions.md
│
├── src/cacheblend/      # The actual implementation.
├── tests/               # Pytest suites, one per phase.
├── benchmarks/          # End-to-end evaluation scripts.
│
├── scripts/
│   ├── send_report.py   # Gmail SMTP email reporter.
│   ├── verify_phase.py  # Auto-checks a phase's acceptance criteria.
│   └── update_status.py # Updates reports/STATUS.md.
│
├── reports/             # Per-phase reports (markdown).
│   ├── STATUS.md        # Current phase tracking.
│   └── TEMPLATE.md      # Report template.
│
└── .github/workflows/   # CI + automatic email on phase tag push.
```

## Phase workflow

```
┌─────────────────┐
│ User: "Do       │
│  Phase N"       │
└────────┬────────┘
         ▼
┌─────────────────────────────────────────┐
│ Claude Code:                            │
│  1. Read GOAL.md, PHASES.md, STATUS.md  │
│  2. Read tasks/phase-N-*.md             │
│  3. Implement                           │
│  4. Run tests + verify_phase.py         │
│  5. Write reports/phase-N-report.md     │
│  6. send_report.py → email              │
└────────┬────────────────────────────────┘
         ▼
┌─────────────────────────────────────────┐
│ User reads email, asks Claude (chat)    │
│ to draft prompt for Phase N+1           │
└─────────────────────────────────────────┘
```

## License

Code: TBD. Paper: see [original CacheBlend paper](https://arxiv.org/abs/2405.16444).
