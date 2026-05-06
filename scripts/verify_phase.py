#!/usr/bin/env python3
"""
Verify that a phase's acceptance criteria are met.

Usage:
    python scripts/verify_phase.py --phase 1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> tuple[int, str]:
    """Returns (returncode, combined stdout+stderr)."""
    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True
    )
    return proc.returncode, proc.stdout + proc.stderr


def check_files_exist(paths: list[Path]) -> tuple[bool, list[str]]:
    missing = [str(p) for p in paths if not p.exists()]
    return (len(missing) == 0), missing


PHASES = {
    0: {
        "name": "Setup & Analysis",
        "files": [
            REPO_ROOT / "docs" / "paper-summary.md",
            REPO_ROOT / "docs" / "lmcache-analysis.md",
            REPO_ROOT / "src" / "cacheblend" / "__init__.py",
        ],
        "tests": None,  # Phase 0 has no automated tests
        "extra_checks": [
            ("Email pipeline dry-run",
             ["python", "scripts/send_report.py", "--phase", "0", "--dry-run"]),
        ],
    },
    1: {
        "name": "Layerwise Forward",
        "files": [
            REPO_ROOT / "src" / "cacheblend" / "model.py",
        ],
        "tests": ["pytest", "tests/test_layerwise.py", "-v"],
        "extra_checks": [],
    },
    2: {
        "name": "KV Storage & Full Reuse",
        "files": [
            REPO_ROOT / "src" / "cacheblend" / "kv_store.py",
            REPO_ROOT / "src" / "cacheblend" / "rope.py",
            REPO_ROOT / "src" / "cacheblend" / "fusor.py",
            REPO_ROOT / "src" / "cacheblend" / "chunker.py",
        ],
        "tests": ["pytest", "tests/test_kv_reuse.py", "-v"],
        "extra_checks": [],
    },
    3: {
        "name": "Selective KV Recompute",
        "files": [
            REPO_ROOT / "src" / "cacheblend" / "hkvd.py",
        ],
        "tests": ["pytest", "tests/test_selective.py", "-v"],
        "extra_checks": [],
    },
    4: {
        "name": "Pipelining",
        "files": [
            REPO_ROOT / "src" / "cacheblend" / "controller.py",
            REPO_ROOT / "benchmarks" / "ttft.py",
        ],
        "tests": ["pytest", "tests/test_pipeline.py", "-v"],
        "extra_checks": [],
    },
    5: {
        "name": "End-to-end Evaluation",
        "files": [
            REPO_ROOT / "benchmarks" / "run_benchmark.py",
        ],
        "tests": ["pytest", "tests/test_e2e.py", "-v", "-m", "slow"],
        "extra_checks": [],
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, required=True, choices=list(PHASES.keys()))
    args = parser.parse_args()

    spec = PHASES[args.phase]
    print(f"\n=== Verifying Phase {args.phase}: {spec['name']} ===\n")

    all_ok = True

    # 1. Files exist
    ok, missing = check_files_exist(spec["files"])
    if ok:
        print(f"✅ Required files present ({len(spec['files'])} files)")
    else:
        print(f"❌ Missing files:")
        for m in missing:
            print(f"   - {m}")
        all_ok = False

    # 2. Tests pass
    if spec["tests"]:
        print(f"\nRunning: {' '.join(spec['tests'])}")
        rc, out = run(spec["tests"])
        if rc == 0:
            print(f"✅ Tests passed")
        else:
            print(f"❌ Tests failed (returncode {rc})")
            print(out[-2000:])  # last 2KB of output
            all_ok = False

    # 3. Extra checks
    for label, cmd in spec["extra_checks"]:
        print(f"\nRunning: {label} — {' '.join(cmd)}")
        rc, out = run(cmd)
        if rc == 0:
            print(f"✅ {label} passed")
        else:
            print(f"❌ {label} failed (returncode {rc})")
            print(out[-1000:])
            all_ok = False

    print()
    if all_ok:
        print(f"🎉 Phase {args.phase} verified.")
        return 0
    else:
        print(f"💥 Phase {args.phase} NOT yet complete. See errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
