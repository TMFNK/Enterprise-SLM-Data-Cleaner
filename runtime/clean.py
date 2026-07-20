"""
clean.py: v1 runtime. ONE fine-tuned model normalizes a record; a cheap
validation gate checks it; the deterministic algorithm is the safety net.

No escalation router, no second model. Flow per record:

    record --> fine-tuned Qwen3-0.6B (llama.cpp, grammar-constrained JSON)
           --> validate against convention_spec (schema + rules)
           --> if invalid: fall back to normalize_record() (the algorithm)
                           or flag needs_review

Every decision is appended to the audit log; flagged records also land in the
review queue (runtime/audit.py, worked off with runtime/review.py). Model
confidence below --min-confidence is flagged even when the rule checks pass.

Why an LLM at all if the algorithm exists? The algorithm only covers the rules we
wrote. The model is there to generalize to messiness the rules DON'T cover
(novel typos, unseen aliases, fuzzy city/name matches). The algorithm is the
guardrail for the known cases; eval measures how much the model adds on top.

Run `python clean.py` for an offline demo (algorithm only, no server needed).
Run `python clean.py --live` to use the served fine-tuned model.
"""
from __future__ import annotations
import os
import sys
import json
import argparse

# make convention_spec (in core/) importable when run from anywhere
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
from convention_spec import normalize_record, rule_violations
from llama_client import call_model, require_server

_SERVER_HINT = """
Cannot reach the model server at http://localhost:{port}.

Start it in a SEPARATE terminal and wait until it prints that it is listening:

    make serve

If you picked a custom port, use it on both sides: make serve PORT={port}.
The first run also downloads the model, so give it a minute. Then re-run this.
"""


def clean_record(record: dict, use_model: bool = True,
                 min_confidence: float = 0.9,
                 model_name: str = "qwen3-0.6b-cleaner",
                 port: int = 8080) -> dict:
    """Return {result, source, needs_review, violations}."""
    if use_model:
        obj = call_model(record, model_name=model_name, port=port)
        violations = rule_violations("mdm_record", obj) if obj else ["no valid JSON"]
        if obj and not violations:
            conf = obj.get("confidence", 1.0)
            if conf < min_confidence:
                # Rules pass but the model is unsure: never silently through.
                return {"result": obj, "source": "model", "needs_review": True,
                        "violations": [f"confidence {conf} below threshold {min_confidence}"]}
            return {"result": obj, "source": "model", "needs_review": False, "violations": []}
        # Safety net: deterministic algorithm covers the rule-defined fields.
        fixed, _ = normalize_record(record)
        return {"result": fixed, "source": "algorithm_fallback",
                "needs_review": True, "violations": violations}
    # Algorithm-only mode (no model / offline).
    fixed, _ = normalize_record(record)
    violations = rule_violations("mdm_record", fixed)
    return {"result": fixed, "source": "algorithm",
            "needs_review": bool(violations), "violations": violations}


if __name__ == "__main__":
    from audit import AuditLog

    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="use the served fine-tuned model")
    ap.add_argument("--port", type=int, default=8080,
                    help="port of the llama.cpp server (match make serve PORT=...)")
    ap.add_argument("--model-name", default="qwen3-0.6b-cleaner",
                    help="model name sent to llama.cpp (match make ALIAS=...)")
    ap.add_argument("--batch", help="clean a JSONL file of records (one JSON object per line)")
    ap.add_argument("--out", help="write cleaned records to this JSONL file (with --batch)")
    ap.add_argument("--audit-dir", default="audit",
                    help="where the append-only audit log + review queue live")
    ap.add_argument("--min-confidence", type=float, default=0.9,
                    help="model confidence below this goes to manual review")
    ap.add_argument("--model-file", help="path to the served GGUF, to hash into the audit log")
    args = ap.parse_args()
    if args.live:
        require_server(args.port, _SERVER_HINT)

    log = AuditLog(args.audit_dir, model=args.model_name if args.live else "algorithm",
                   model_file=args.model_file)

    if args.batch:
        records = [json.loads(l) for l in open(args.batch, encoding="utf-8")]
        flagged, cleaned = 0, []
        for rec in records:
            out = clean_record(rec, use_model=args.live,
                               min_confidence=args.min_confidence,
                               model_name=args.model_name, port=args.port)
            log.record(rec, out)
            flagged += int(out["needs_review"])
            cleaned.append(out["result"])
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                for c in cleaned:
                    fh.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"cleaned {len(records)} records: {len(records) - flagged} ok, "
              f"{flagged} -> review queue")
        print(f"audit log: {log.log_path}"
              + (f", output: {args.out}" if args.out else ""))
        if flagged:
            print("work off the queue with: python runtime/review.py list")
        sys.exit(0)

    demo = {"recordId": "v-1001", "recordType": "vendor", "name1": "  Muster  Handels ",
            "legalForm": "mbH", "city": "München ", "country": "Germany",
            "iban": "de89 3704 0044 0532 0130 00", "email": "INFO@Muster.DE",
            "currency": "€", "baseUnit": "pcs", "status": "aktiv",
            "validFrom": "01.03.2024", "amount": "1.234,56"}

    out = clean_record(demo, use_model=args.live, min_confidence=args.min_confidence,
                       model_name=args.model_name, port=args.port)
    log.record(demo, out)
    print(f"source={out['source']} needs_review={out['needs_review']}")
    if out["violations"]:
        print("model violations:", out["violations"])
    print(json.dumps(out["result"], ensure_ascii=False, indent=2))
