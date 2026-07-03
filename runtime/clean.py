"""
clean.py: v1 runtime. ONE fine-tuned model normalizes a record; a cheap
validation gate checks it; the deterministic algorithm is the safety net.

No escalation router, no second model. Flow per record:

    record --> fine-tuned Qwen3-0.6B (llama.cpp, grammar-constrained JSON)
           --> validate against convention_spec (schema + rules)
           --> if invalid: fall back to normalize_record() (the algorithm)
                           or flag needs_review

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
import convention_spec as spec
from convention_spec import normalize_record, rule_violations

try:
    import requests
except ImportError:
    requests = None

MODEL_URL = "http://localhost:8080/v1/chat/completions"   # port set via --port
MODEL_NAME = "qwen3-0.6b-cleaner"

_SERVER_HINT = """
Cannot reach the model server at http://localhost:{port}.

Start it in a SEPARATE terminal and wait until it prints that it is listening:

    make serve

If you picked a custom port, use it on both sides: make serve PORT={port}.
The first run also downloads the model, so give it a minute. Then re-run this.
"""


def require_server(port: int):
    """Fail early with a friendly message if the model server is not up."""
    if requests is None:
        sys.exit("The `requests` package is missing. Run: make setup")
    models_url = MODEL_URL.rsplit("/", 2)[0] + "/models"  # .../v1/models
    try:
        requests.get(models_url, timeout=3)
    except requests.exceptions.RequestException:
        sys.exit(_SERVER_HINT.format(port=port))


def _call_model(record: dict) -> dict | None:
    if requests is None:
        raise RuntimeError("`requests` not installed; --live needs it")
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": spec.system_prompt("mdm_record")},
            {"role": "user", "content": json.dumps(record, ensure_ascii=False)},
        ],
        "temperature": 0,
        # Grammar-constrain to the record schema so output is always valid JSON.
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "mdm_record", "schema": spec.BLOCK_SCHEMAS["mdm_record"]},
        },
    }
    r = requests.post(MODEL_URL, json=payload, timeout=120)
    r.raise_for_status()
    try:
        return json.loads(r.json()["choices"][0]["message"]["content"])
    except (json.JSONDecodeError, KeyError):
        return None


def clean_record(record: dict, use_model: bool = True) -> dict:
    """Return {result, source, needs_review, violations}."""
    if use_model:
        obj = _call_model(record)
        violations = rule_violations("mdm_record", obj) if obj else ["no valid JSON"]
        if obj and not violations:
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="use the served fine-tuned model")
    ap.add_argument("--port", type=int, default=8080,
                    help="port of the llama.cpp server (match make serve PORT=...)")
    args = ap.parse_args()
    MODEL_URL = f"http://localhost:{args.port}/v1/chat/completions"
    if args.live:
        require_server(args.port)

    demo = {"recordId": "v-1001", "recordType": "vendor", "name1": "  Muster  Handels ",
            "legalForm": "mbH", "city": "München ", "country": "Germany",
            "iban": "de89 3704 0044 0532 0130 00", "email": "INFO@Muster.DE",
            "currency": "€", "baseUnit": "pcs", "status": "aktiv",
            "validFrom": "01.03.2024", "amount": "1.234,56"}

    out = clean_record(demo, use_model=args.live)
    print(f"source={out['source']} needs_review={out['needs_review']}")
    if out["violations"]:
        print("model violations:", out["violations"])
    print(json.dumps(out["result"], ensure_ascii=False, indent=2))
