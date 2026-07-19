#!/usr/bin/env python3
"""Build pinned gold + unseen-holdout fixtures (run once; commit the output).

Gold is sacred: synth/generate.py must never write here. Re-run this script
only when intentionally refreshing the pinned set (bump MANIFEST).
"""
from __future__ import annotations
import hashlib
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))
sys.path.insert(0, os.path.join(ROOT, "synth"))

from convention_spec import normalize_record, rule_violations, system_prompt, CONVENTION_PATH
from generate import clean_record, corrupt_record

FIXTURES = os.path.join(ROOT, "fixtures")
GOLD_N = 100
GOLD_SEED = 42
UNSEEN_SEED = 99
UNSEEN_N = 40


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _example(messy: dict, category: str | None = None) -> dict:
    target, changes = normalize_record(messy)
    target["confidence"] = 1.0 if not rule_violations("mdm_record", target) else 0.6
    target["changes"] = changes
    row = {"messages": [
        {"role": "system", "content": system_prompt("mdm_record")},
        {"role": "user", "content": json.dumps(messy, ensure_ascii=False)},
        {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
    ]}
    if category:
        row["category"] = category
    return row


def corrupt_unseen(clean: dict, rng: random.Random) -> dict:
    """Corruption family NOT used in synth/generate.corrupt_record.

    OCR-ish / glue artifacts: mid-token spaces, duplicated punctuation,
    glued currency symbols. Excluded from training on purpose.
    """
    m = dict(clean)
    if "name1" in m and isinstance(m["name1"], str):
        s = m["name1"]
        if len(s) > 4:
            i = rng.randint(1, len(s) - 2)
            m["name1"] = s[:i] + " " + s[i:]
        m["name1"] = m["name1"].replace(" ", "  ", 1) if " " in m["name1"] else m["name1"]
    if "country" in m and isinstance(m["country"], str) and rng.random() < 0.8:
        c = m["country"]
        m["country"] = c[0] + " " + c[1:] if len(c) == 2 else (" ".join(c))
    if "legalForm" in m and isinstance(m["legalForm"], str) and rng.random() < 0.7:
        m["legalForm"] = m["legalForm"] + m["legalForm"][-1]  # duplicated last char
    if "email" in m and isinstance(m["email"], str) and rng.random() < 0.5:
        m["email"] = m["email"].replace("@", "@@")
    if "amount" in m and rng.random() < 0.6:
        m["amount"] = f"EUREUR {m['amount']}"
    if "iban" in m and isinstance(m["iban"], str) and rng.random() < 0.5:
        v = m["iban"]
        m["iban"] = " ".join(list(v[:4])) + v[4:]  # spaced prefix OCR
    return m


def main() -> None:
    os.makedirs(FIXTURES, exist_ok=True)
    os.makedirs(os.path.join(FIXTURES, "real", "local"), exist_ok=True)

    rtypes = ["vendor", "customer", "material", "costCenter", "glAccount"]
    weights = [4, 3, 3, 1, 1]

    rng = random.Random(GOLD_SEED)
    gold = []
    for _ in range(GOLD_N):
        rtype = rng.choices(rtypes, weights=weights)[0]
        clean = clean_record(rtype, rng)
        messy = corrupt_record(clean, rng)
        gold.append(_example(messy))

    gold_path = os.path.join(FIXTURES, "gold.jsonl")
    with open(gold_path, "w", encoding="utf-8") as fh:
        for row in gold:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    rng_u = random.Random(UNSEEN_SEED)
    unseen = []
    for _ in range(UNSEEN_N):
        rtype = rng_u.choices(rtypes, weights=weights)[0]
        clean = clean_record(rtype, rng_u)
        messy = corrupt_unseen(clean, rng_u)
        unseen.append(_example(messy, category="unseen"))

    unseen_path = os.path.join(FIXTURES, "holdout_unseen_noise.jsonl")
    with open(unseen_path, "w", encoding="utf-8") as fh:
        for row in unseen:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    conv_hash = _sha256(CONVENTION_PATH)
    manifest = os.path.join(FIXTURES, "MANIFEST.md")
    with open(manifest, "w", encoding="utf-8") as fh:
        fh.write("# Fixtures manifest\n\n")
        fh.write(f"- convention_path: `{CONVENTION_PATH}`\n")
        fh.write(f"- convention_sha256: `{conv_hash}`\n")
        fh.write(f"- gold.jsonl: {GOLD_N} rows, seed={GOLD_SEED}\n")
        fh.write(f"- holdout_unseen_noise.jsonl: {UNSEEN_N} rows, seed={UNSEEN_SEED}\n")
        fh.write("- gold is sacred: never overwritten by `synth/generate.py`\n")
        fh.write("- unseen corruption family is excluded from training corruptors\n")

    print(f"wrote {gold_path} ({GOLD_N})")
    print(f"wrote {unseen_path} ({UNSEEN_N})")
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
