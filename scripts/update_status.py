#!/usr/bin/env python3
"""
Update reports/STATUS.md with the current phase status.

Usage:
    python scripts/update_status.py --phase 1 --status completed
    python scripts/update_status.py --phase 2 --status in_progress
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STATUS_PATH = REPO_ROOT / "reports" / "STATUS.md"

PHASES = {
    0: "Setup & Analysis",
    1: "Layerwise Forward",
    2: "KV Storage & Full Reuse",
    3: "Selective KV Recompute",
    4: "Pipelining",
    5: "End-to-end Evaluation",
}


def render(state: dict) -> str:
    lines = ["# Project Status\n"]
    lines.append(f"_Last updated: {state.get('updated', 'never')}_\n")
    lines.append("| Phase | Name | Status |")
    lines.append("|---|---|---|")
    for p in range(6):
        status = state.get(f"phase_{p}", "not started")
        emoji = {"completed": "✅", "in_progress": "🔄", "blocked": "🛑", "not started": "⬜"}.get(status, "❓")
        lines.append(f"| {p} | {PHASES[p]} | {emoji} {status} |")
    lines.append("")
    if "notes" in state:
        lines.append(f"## Notes\n{state['notes']}\n")
    return "\n".join(lines)


def parse(text: str) -> dict:
    state = {}
    for line in text.splitlines():
        if line.startswith("| "):
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) >= 3 and parts[0].isdigit():
                phase = int(parts[0])
                # status is "<emoji> <text>", strip emoji
                status_text = parts[2].split(" ", 1)[-1] if " " in parts[2] else parts[2]
                state[f"phase_{phase}"] = status_text
        elif line.startswith("_Last updated"):
            state["updated"] = line.split(": ", 1)[1].rstrip("_")
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, required=True, choices=list(PHASES.keys()))
    parser.add_argument("--status", required=True,
                        choices=["completed", "in_progress", "blocked", "not started"])
    parser.add_argument("--notes", default=None, help="Optional notes to append.")
    args = parser.parse_args()

    if STATUS_PATH.exists():
        state = parse(STATUS_PATH.read_text())
    else:
        state = {}

    state[f"phase_{args.phase}"] = args.status
    state["updated"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if args.notes:
        state["notes"] = args.notes

    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(render(state))
    print(f"✅ Updated {STATUS_PATH.relative_to(REPO_ROOT)}: phase {args.phase} → {args.status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
