"""
eval/evaluate.py: score a cleaner against the held-out test set.

Two modes:
  --algorithm  run the deterministic normalizer as the "predictor". This checks the
            dataset is internally consistent (should be ~100%). Sanity, not a
            real model eval.
  --live    call the served model (llama.cpp OpenAI API) and score its output.
            This is the real before/after eval for a fine-tune.

Metrics: valid-JSON rate, exact-record match, per-field accuracy.

Usage:
    python eval/evaluate.py --data data/test.jsonl --algorithm
    python eval/evaluate.py --data data/test.jsonl --live
"""
from __future__ import annotations
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
import convention_spec as spec
from convention_spec import normalize_record

try:
    import requests
except ImportError:
    requests = None

MODEL_URL = "http://localhost:8080/v1/chat/completions"  # port set via --port
MODEL_NAME = "qwen3-0.6b-cleaner"

_SERVER_HINT = """
Cannot reach the model server at http://localhost:{port}.

The server has to be running in a SEPARATE terminal before you score it. Start
one of these and wait until it prints that it is listening:

    make baseline-serve    (the untrained model, for your 'before' score)
    make serve             (your fine-tuned model, for your 'after' score)

If you picked a custom port, use it on both sides: make serve PORT={port}.
The first baseline-serve run also downloads the model (about 600 MB), so give
it a minute. Then re-run this command in your second terminal.
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


# fields whose values are floats/None -> compare loosely
def _eq(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) < 1e-6
        except (TypeError, ValueError):
            return a == b
    return a == b


def predict_algorithm(messy: dict) -> dict:
    target, changes = normalize_record(messy)
    target["confidence"] = 1.0
    target["changes"] = changes
    return target


def predict_live(messy: dict) -> dict | None:
    if requests is None:
        raise RuntimeError("`requests` needed for --live")
    payload = {
        "model": MODEL_NAME, "temperature": 0,
        "messages": [
            {"role": "system", "content": spec.system_prompt("mdm_record")},
            {"role": "user", "content": json.dumps(messy, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_schema", "json_schema":
                            {"name": "mdm_record", "schema": spec.BLOCK_SCHEMAS["mdm_record"]}},
    }
    r = requests.post(MODEL_URL, json=payload, timeout=120)
    r.raise_for_status()
    try:
        return json.loads(r.json()["choices"][0]["message"]["content"])
    except (json.JSONDecodeError, KeyError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--algorithm", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--port", type=int, default=8080,
                    help="port of the llama.cpp server (match make serve PORT=...)")
    args = ap.parse_args()
    global MODEL_URL
    MODEL_URL = f"http://localhost:{args.port}/v1/chat/completions"
    predict = predict_live if args.live else predict_algorithm
    if args.live:
        require_server(args.port)

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]

    valid_json = exact = 0
    field_hits = field_total = 0
    compared_fields = set(spec.FIELD_REGISTRY)  # ignore confidence/changes

    for ex in rows:
        msgs = {m["role"]: m["content"] for m in ex["messages"]}
        messy = json.loads(msgs["user"])
        gold = json.loads(msgs["assistant"])
        pred = predict(messy)
        if pred is None:
            continue
        valid_json += 1
        rec_ok = True
        for f in compared_fields:
            if f in gold:
                field_total += 1
                if _eq(pred.get(f), gold.get(f)):
                    field_hits += 1
                else:
                    rec_ok = False
        exact += int(rec_ok)

    n = len(rows)
    print(f"mode           : {'live model' if args.live else 'algorithm (sanity)'}")
    print(f"examples       : {n}")
    print(f"valid JSON     : {valid_json/n:6.1%}")
    print(f"exact record   : {exact/n:6.1%}")
    print(f"field accuracy : {field_hits/max(field_total,1):6.1%}  ({field_hits}/{field_total})")


if __name__ == "__main__":
    main()
