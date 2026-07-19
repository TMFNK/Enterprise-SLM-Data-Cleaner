"""
eval/evaluate.py: score a cleaner against a fixture or generated split.

Two modes:
  --algorithm  run the deterministic normalizer as the "predictor". This checks the
            dataset is internally consistent (should be ~100%). Also the permanent
            **oracle peer** in eval reports.
  --live    call the served model (llama.cpp OpenAI API) and score its output.
            This is the real before/after eval for a fine-tune.

Metrics: valid-JSON rate, exact-record match, per-field accuracy, per-field support.
Optional --report writes a markdown stub for the oracle · base · FT table.

Usage:
    python eval/evaluate.py --data fixtures/gold.jsonl --algorithm
    python eval/evaluate.py --data fixtures/gold.jsonl --algorithm --report reports/oracle-gold.md --label oracle
    python eval/evaluate.py --data data/test.jsonl --live
"""
from __future__ import annotations
import os
import sys
import json
import argparse
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
import convention_spec as spec
from convention_spec import normalize_record

try:
    import requests
except ImportError:
    requests = None

MODEL_URL = "http://localhost:8080/v1/chat/completions"

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
    if requests is None:
        sys.exit("The `requests` package is missing. Run: make setup")
    models_url = MODEL_URL.rsplit("/", 2)[0] + "/models"
    try:
        requests.get(models_url, timeout=3)
    except requests.exceptions.RequestException:
        sys.exit(_SERVER_HINT.format(port=port))


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


def predict_live(messy: dict, model_name: str = "qwen3-0.6b-cleaner") -> dict | None:
    if requests is None:
        raise RuntimeError("`requests` needed for --live")
    payload = {
        "model": model_name, "temperature": 0,
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


def _write_report(path: str, label: str, mode: str, data_path: str,
                  n: int, valid_json: int, exact: int,
                  field_hits: int, field_total: int,
                  by_cat: dict, field_support: Counter, field_hits_map: Counter) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    score = 100.0 * field_hits / max(field_total, 1)
    validity = 100.0 * valid_json / max(n, 1)
    lines = [
        f"# Eval report — {label}",
        "",
        f"- ts: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
        f"- mode: `{mode}`",
        f"- label: `{label}`",
        f"- data: `{data_path}`",
        f"- convention: `{spec.CONVENTION_PATH}`",
        f"- examples: {n}",
        f"- validity: {validity:.1f}%",
        f"- exact record: {100.0 * exact / max(n, 1):.1f}%",
        f"- field accuracy: {score:.1f}% ({field_hits}/{field_total})",
        "",
        "## Peer table (fill base / fine-tuned when scored)",
        "",
        "| System | Validity | Field accuracy |",
        "|---|---|---|",
        f"| {label} | {validity:.1f}% | {score:.1f}% |",
        "| base SLM | | |",
        "| fine-tuned SLM | | |",
        "",
        "## Per-category field accuracy",
        "",
    ]
    if by_cat:
        lines.append("| Category | Accuracy | Hits |")
        lines.append("|---|---|---|")
        for cat, (hits, total) in sorted(by_cat.items()):
            lines.append(f"| {cat} | {100.0 * hits / max(total, 1):.1f}% | {hits}/{total} |")
    else:
        lines.append("_No category labels in this fixture._")
    lines.extend(["", "## Per-field support (gold fields compared)", ""])
    lines.append("| Field | Support | Hits | Accuracy |")
    lines.append("|---|---|---|---|")
    for f, total in sorted(field_support.items()):
        hits = field_hits_map[f]
        lines.append(f"| {f} | {total} | {hits} | {100.0 * hits / max(total, 1):.1f}% |")
    lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"report written: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--algorithm", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--port", type=int, default=8080,
                    help="port of the llama.cpp server (match make serve PORT=...)")
    ap.add_argument("--model-name", default="qwen3-0.6b-cleaner",
                    help="model name sent to llama.cpp (match make ALIAS=...)")
    ap.add_argument("--min-score", type=float, default=None,
                    help="exit non-zero if field accuracy (%%) is below this (the eval gate)")
    ap.add_argument("--report", default=None,
                    help="write a markdown report to this path")
    ap.add_argument("--label", default=None,
                    help="row label for the report (oracle / base / fine-tuned)")
    args = ap.parse_args()
    global MODEL_URL
    MODEL_URL = f"http://localhost:{args.port}/v1/chat/completions"
    if args.live:
        predict = lambda m: predict_live(m, model_name=args.model_name)
        mode = "live model"
        default_label = "fine-tuned SLM"
    else:
        predict = predict_algorithm
        mode = "algorithm (oracle)"
        default_label = "oracle"
    label = args.label or default_label
    if args.live:
        require_server(args.port)

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]

    valid_json = exact = 0
    field_hits = field_total = 0
    by_cat: dict[str, list[int]] = {}
    field_support: Counter = Counter()
    field_hits_map: Counter = Counter()
    compared_fields = set(spec.FIELD_REGISTRY)
    review_flagged = 0

    for ex in rows:
        msgs = {m["role"]: m["content"] for m in ex["messages"]}
        messy = json.loads(msgs["user"])
        gold = json.loads(msgs["assistant"])
        cat = ex.get("category")
        pred = predict(messy)
        if pred is None:
            continue
        valid_json += 1
        conf = pred.get("confidence")
        if isinstance(conf, (int, float)) and conf < 0.9:
            review_flagged += 1
        rec_ok = True
        for f in compared_fields:
            if f in gold:
                field_total += 1
                field_support[f] += 1
                hit = _eq(pred.get(f), gold.get(f))
                field_hits += int(hit)
                field_hits_map[f] += int(hit)
                if cat:
                    by_cat.setdefault(cat, [0, 0])
                    by_cat[cat][0] += int(hit)
                    by_cat[cat][1] += 1
                if not hit:
                    rec_ok = False
        exact += int(rec_ok)

    n = len(rows)
    score = 100.0 * field_hits / max(field_total, 1)
    print(f"mode           : {mode}")
    print(f"label          : {label}")
    print(f"examples       : {n}")
    print(f"valid JSON     : {valid_json/n:6.1%}")
    print(f"exact record   : {exact/n:6.1%}")
    print(f"field accuracy : {field_hits/max(field_total,1):6.1%}  ({field_hits}/{field_total})")
    print(f"low-conf share : {review_flagged/max(n,1):6.1%}  (confidence < 0.9; ops review-rate signal)")
    for cat, (hits, total) in sorted(by_cat.items()):
        print(f"  {cat:13s}: {hits/max(total,1):6.1%}  ({hits}/{total})")
    if args.report:
        _write_report(args.report, label, mode, args.data, n, valid_json, exact,
                      field_hits, field_total, by_cat, field_support, field_hits_map)
    if args.min_score is not None and score < args.min_score:
        print(f"EVAL GATE FAILED: field accuracy {score:.1f}% < required {args.min_score:.1f}%")
        sys.exit(1)


if __name__ == "__main__":
    main()
