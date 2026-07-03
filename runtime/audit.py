"""
runtime/audit.py: append-only audit trail + review queue.

Every cleaned record produces ONE line in the audit log (JSONL): timestamp,
input, output, source, violations, confidence, model identity, and the
convention spec (path + sha256) that was in force. Records flagged
`needs_review` are ALSO appended to the review queue, to be worked off with
runtime/review.py. Nothing is ever mutated or deleted: resolutions are
separate append-only entries.

Files (under the audit dir, default `audit/`):
    audit.jsonl               every cleaning decision, append-only
    review-queue.jsonl        flagged records awaiting manual review
    review-resolutions.jsonl  written by review.py, one entry per resolved item
"""
from __future__ import annotations
import os
import sys
import json
import hashlib
from datetime import datetime, timezone

# make convention_spec (in core/) importable when run from anywhere
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
import convention_spec as spec


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convention_fingerprint() -> dict:
    return {"path": spec.CONVENTION_PATH,
            "sha256": sha256_file(spec.CONVENTION_PATH)}


def _entry_id(entry: dict) -> str:
    blob = json.dumps(entry, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def _append(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


class AuditLog:
    def __init__(self, root: str = "audit", model: str = "algorithm",
                 model_file: str | None = None):
        os.makedirs(root, exist_ok=True)
        self.log_path = os.path.join(root, "audit.jsonl")
        self.queue_path = os.path.join(root, "review-queue.jsonl")
        self.model = model
        # hash the served model file once, if the caller can point at it
        self.model_sha256 = sha256_file(model_file) if model_file else None
        self.convention = convention_fingerprint()

    def record(self, input_record: dict, result: dict) -> dict:
        """result is clean_record()'s dict: result/source/needs_review/violations."""
        out = result["result"]
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "input": input_record,
            "output": out,
            "source": result["source"],
            "needs_review": result["needs_review"],
            "violations": result["violations"],
            "confidence": out.get("confidence"),
            "changes": out.get("changes"),
            "model": self.model,
            "model_sha256": self.model_sha256,
            "convention": self.convention,
        }
        entry["id"] = _entry_id(entry)
        _append(self.log_path, entry)
        if entry["needs_review"]:
            _append(self.queue_path, entry)
        return entry
