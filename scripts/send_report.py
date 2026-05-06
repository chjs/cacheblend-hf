#!/usr/bin/env python3
"""
Send a phase report via email (Gmail SMTP).

Usage:
    python scripts/send_report.py --phase 1
    python scripts/send_report.py --phase 1 --dry-run

Environment variables (.env or shell):
    GMAIL_ADDRESS         your Gmail address (from)
    GMAIL_APP_PASSWORD    Gmail app password (NOT regular password)
                          https://myaccount.google.com/apppasswords
    REPORT_EMAIL_TO       recipient email (default: ch.jungsik@gmail.com)
"""
from __future__ import annotations

import argparse
import os
import smtplib
import sys
from email.message import EmailMessage
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
DEFAULT_TO = "ch.jungsik@gmail.com"


def find_report(phase: int) -> Path:
    candidates = sorted(REPORTS_DIR.glob(f"phase-{phase}-report*.md"))
    if not candidates:
        raise FileNotFoundError(
            f"No report found for phase {phase}. "
            f"Expected at reports/phase-{phase}-report.md"
        )
    return candidates[-1]


def collect_attachments(phase: int) -> list[Path]:
    """Optional: include test logs, plots, etc."""
    attachments_dir = REPORTS_DIR / f"phase-{phase}-attachments"
    if not attachments_dir.exists():
        return []
    return [p for p in attachments_dir.iterdir() if p.is_file()]


def build_message(
    phase: int,
    report_text: str,
    sender: str,
    to: str,
    attachments: list[Path],
) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = f"[CacheBlend-HF] Phase {phase} report"
    msg["From"] = sender
    msg["To"] = to

    body = (
        f"Phase {phase} report below.\n"
        f"This is an automated message from the cacheblend-hf harness.\n"
        f"---\n\n"
        f"{report_text}"
    )
    msg.set_content(body)

    for path in attachments:
        ctype, _ = "application", "octet-stream"
        if path.suffix in {".png", ".jpg", ".jpeg"}:
            ctype, subtype = "image", path.suffix.lstrip(".").replace("jpg", "jpeg")
        elif path.suffix in {".txt", ".md", ".log", ".json", ".csv"}:
            ctype, subtype = "text", "plain"
        else:
            ctype, subtype = "application", "octet-stream"
        msg.add_attachment(
            path.read_bytes(),
            maintype=ctype,
            subtype=subtype,
            filename=path.name,
        )
    return msg


def send(msg: EmailMessage, sender: str, app_password: str):
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, app_password)
        server.send_message(msg)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't actually send. Verify config and print preview.")
    parser.add_argument("--to", default=os.environ.get("REPORT_EMAIL_TO", DEFAULT_TO))
    args = parser.parse_args()

    sender = os.environ.get("GMAIL_ADDRESS")
    app_password = os.environ.get("GMAIL_APP_PASSWORD")

    if not sender or not app_password:
        if args.dry_run:
            print("[dry-run] GMAIL_ADDRESS / GMAIL_APP_PASSWORD not set.")
            print("[dry-run] Set them in .env or environment to actually send.")
        else:
            print(
                "ERROR: GMAIL_ADDRESS and GMAIL_APP_PASSWORD must be set.\n"
                "  cp .env.example .env  and fill in the values.\n"
                "  See https://myaccount.google.com/apppasswords",
                file=sys.stderr,
            )
            return 2

    # In dry-run mode, allow phase 0 with no report file (initial setup test)
    if args.phase == 0 and args.dry_run:
        report_text = "(dry-run: phase 0 setup verification — no report needed)"
        attachments = []
    else:
        try:
            report_path = find_report(args.phase)
        except FileNotFoundError as e:
            if args.dry_run:
                print(f"[dry-run] {e}")
                report_text = f"(dry-run: would have read reports/phase-{args.phase}-report.md)"
                attachments = []
            else:
                print(f"ERROR: {e}", file=sys.stderr)
                return 3
        else:
            report_text = report_path.read_text()
            attachments = collect_attachments(args.phase)

    if args.dry_run:
        print(f"[dry-run] Would send phase {args.phase} report to {args.to}")
        print(f"[dry-run] From: {sender or '(not set)'}")
        print(f"[dry-run] Attachments: {[p.name for p in attachments]}")
        print("[dry-run] Body preview (first 500 chars):")
        print(report_text[:500])
        print("[dry-run] OK")
        return 0

    msg = build_message(args.phase, report_text, sender, args.to, attachments)
    try:
        send(msg, sender, app_password)
    except smtplib.SMTPAuthenticationError:
        print(
            "ERROR: SMTP authentication failed. "
            "Verify GMAIL_APP_PASSWORD (not your regular password!).",
            file=sys.stderr,
        )
        return 4
    except Exception as e:
        print(f"ERROR: failed to send: {e}", file=sys.stderr)
        return 5

    print(f"✅ Sent phase {args.phase} report to {args.to}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
