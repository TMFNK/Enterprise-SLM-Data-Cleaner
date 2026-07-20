#!/usr/bin/env python3
"""Fail closed if train JSONL lacks basic recordType / field coverage.

Defaults are CI-safe: each recordType >= min_fraction of rows; at least
min_distinct countries and legalForms among rows that carry those fields.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))
import convention_spec as spec


def _assistant(row: dict) -> dict:
    msgs = {m["role"]: m["content"] for m in row["messages"]}
    return json.loads(msgs["assistant"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--min-fraction", type=float, default=0.05,
                    help="min share per recordType (default 5%)")
    ap.add_argument("--min-distinct-country", type=int, default=3)
    ap.add_argument("--min-distinct-legal", type=int, default=3)
    args = ap.parse_args()

    if not os.path.exists(args.data):
        print(f"BALANCE CHECK FAILED: missing {args.data}")
        return 1

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    if not rows:
        print("BALANCE CHECK FAILED: empty train set")
        return 1

    types = Counter()
    countries = Counter()
    legal = Counter()
    for row in rows:
        out = _assistant(row)
        if out.get("recordType"):
            types[out["recordType"]] += 1
        if out.get("country"):
            countries[out["country"]] += 1
        if out.get("legalForm"):
            legal[out["legalForm"]] += 1

    n = len(rows)
    errors = []
    for rt, c in sorted(types.items()):
        frac = c / n
        if frac < args.min_fraction:
            errors.append(
                f"recordType {rt}: {c}/{n} = {frac:.1%} < {args.min_fraction:.0%}"
            )

    for rt in sorted(spec.RECORD_TYPES):
        if types.get(rt, 0) == 0 and n >= 50:
            errors.append(f"recordType {rt}: missing from train")

    if len(countries) < args.min_distinct_country and n >= 50:
        errors.append(
            f"countries: only {len(countries)} distinct "
            f"(need >= {args.min_distinct_country})"
        )
    if len(legal) < args.min_distinct_legal and n >= 50:
        errors.append(
            f"legalForm: only {len(legal)} distinct "
            f"(need >= {args.min_distinct_legal})"
        )

    print(f"balance-check: {args.data} n={n}")
    print(f"  recordTypes: {dict(types)}")
    print(f"  distinct countries={len(countries)} legalForms={len(legal)}")
    if errors:
        print("BALANCE CHECK FAILED:")
        for e in errors:
            print(f"  {e}")
        return 1
    print("balance-check: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
