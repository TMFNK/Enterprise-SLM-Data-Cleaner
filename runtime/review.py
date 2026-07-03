"""
runtime/review.py: work off the manual review queue, append-only.

    python runtime/review.py list                    show pending items
    python runtime/review.py show <id>               full entry for one item
    python runtime/review.py resolve <id> --outcome approved --note "..."

Resolving never edits the queue: it appends a resolution entry to
review-resolutions.jsonl. Pending = queued ids minus resolved ids, so the
full history of every decision stays on disk.
"""
from __future__ import annotations
import os
import json
import argparse
from datetime import datetime, timezone

OUTCOMES = ("approved", "corrected", "rejected")


def _load(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def _pending(root: str) -> list[dict]:
    queued = _load(os.path.join(root, "review-queue.jsonl"))
    resolved = {r["id"] for r in _load(os.path.join(root, "review-resolutions.jsonl"))}
    return [e for e in queued if e["id"] not in resolved]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["list", "show", "resolve"])
    ap.add_argument("id", nargs="?", help="entry id (for show/resolve)")
    ap.add_argument("--audit-dir", default="audit")
    ap.add_argument("--outcome", choices=OUTCOMES, default="approved")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    if args.command == "list":
        pending = _pending(args.audit_dir)
        if not pending:
            print("review queue: empty")
            return
        print(f"review queue: {len(pending)} pending")
        for e in pending:
            print(f"  {e['id']}  {e['ts']}  source={e['source']}  "
                  f"violations={len(e['violations'])}  conf={e['confidence']}")
        return

    if not args.id:
        raise SystemExit(f"'{args.command}' needs an entry id (see 'list')")
    match = [e for e in _pending(args.audit_dir) if e["id"] == args.id]
    if not match:
        raise SystemExit(f"no pending entry with id {args.id}")
    entry = match[0]

    if args.command == "show":
        print(json.dumps(entry, ensure_ascii=False, indent=2))
        return

    resolution = {
        "id": entry["id"],
        "resolved_ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "outcome": args.outcome,
        "note": args.note,
    }
    path = os.path.join(args.audit_dir, "review-resolutions.jsonl")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(resolution, ensure_ascii=False) + "\n")
    print(f"resolved {entry['id']} as {args.outcome}. "
          f"{len(_pending(args.audit_dir))} still pending.")


if __name__ == "__main__":
    main()
