"""
eval/adversarial.py: the pinned adversarial eval suite.

Unlike the synthetic training data (where the algorithm labels whatever it
produces), every case here has a HARDCODED expected value. Generation asserts
the algorithm still produces exactly that value, so any change to core/ or the
convention spec that alters behavior fails loudly here: this is the regression
gate. The emitted JSONL doubles as a model eval set with per-category scores.

Categories:
    legal_form      German/intl legal-form ambiguity (mbH vs GmbH, ag. vs AG...)
    format          DE vs US dates and amounts, IBAN/VAT/phone/email mangling
    grounding       plausible-but-unknown values that must pass through UNCHANGED;
                    a model that "helpfully corrects" them is hallucinating
                    (regions like Bavaria must NOT become a country code)
    semantic_alias  near-miss variants resolved only by the optional embedding
                    layer (not in the deterministic alias map); emitted only
                    when USE_EMBEDDINGS=1 so the suite never lies

Unseen holdout (separate file, excluded from training corruptors):
    fixtures/holdout_unseen_noise.jsonl; OCR/glue artifacts from
    scripts/build_fixtures.py. Do not mix into data/train.jsonl.

Usage:
    python eval/adversarial.py --out data/adversarial.jsonl
    USE_EMBEDDINGS=1 python eval/adversarial.py --out data/adversarial.jsonl
"""
from __future__ import annotations
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
from convention_spec import normalize_record, rule_violations, system_prompt

# (category, field, messy input, pinned expected output)
CASES = [
    # -- legal-form ambiguity ------------------------------------------------
    ("legal_form", "legalForm", "mbH", "GmbH"),
    ("legal_form", "legalForm", "MBH", "GmbH"),
    ("legal_form", "legalForm", "  Gesellschaft   mit beschränkter Haftung ", "GmbH"),
    ("legal_form", "legalForm", "gmbh.", "GmbH"),
    ("legal_form", "legalForm", "Aktiengesellschaft", "AG"),
    ("legal_form", "legalForm", "ag.", "AG"),
    ("legal_form", "legalForm", "UG (haftungsbeschränkt)", "UG"),
    ("legal_form", "legalForm", "unternehmergesellschaft", "UG"),
    ("legal_form", "legalForm", "LIMITED", "Ltd"),
    ("legal_form", "legalForm", "ltd.", "Ltd"),
    ("legal_form", "legalForm", "Incorporated", "Inc"),
    ("legal_form", "legalForm", "GmbH & Co. KG", "GmbH & Co. KG"),
    # -- format hell: dates, amounts, identifiers ----------------------------
    ("format", "validFrom", "31.12.2024", "2024-12-31"),
    ("format", "validFrom", "01/02/2023", "2023-02-01"),   # DD/MM, not US MM/DD
    ("format", "validFrom", "2024/03/15", "2024-03-15"),
    ("format", "validFrom", "05-04-2022", "2022-04-05"),
    ("format", "validFrom", "2024-03-01", "2024-03-01"),
    ("format", "amount", "1.234,56", 1234.56),             # German decimal
    ("format", "amount", "1,234.56", 1234.56),             # US decimal
    ("format", "amount", "1.234.567,89", 1234567.89),
    ("format", "amount", "EUR 99,90", 99.9),
    ("format", "amount", "-999", None),                    # sentinel -> null
    ("format", "amount", "n/a", None),                     # empty token -> null
    ("format", "iban", "de89 3704 0044 0532 0130 00", "DE89370400440532013000"),
    ("format", "vatId", "de 811.569-869", "DE811569869"),
    ("format", "phone", "0049 (30) 12345", "+493012345"),
    ("format", "email", "Info@FIRMA.de", "info@firma.de"),
    ("format", "country", "  gErMaNy ", "DE"),
    ("format", "country", "BRD", "DE"),
    ("format", "country", "Nederland", "NL"),
    ("format", "country", "Tyskland", "DE"),
    ("format", "currency", "euro", "EUR"),
    ("format", "status", "FREIGEGEBEN", "active"),
    ("format", "status", "currently active", "active"),
    ("format", "status", "not active", "inactive"),
    ("format", "baseUnit", "Stk", "PCE"),
    ("format", "name1", "  Muster   Handels  GmbH ", "Muster Handels GmbH"),
    ("format", "legalForm", "GmbH (Gesellschaft mit beschränkter Haftung)", "GmbH"),
    ("format", "legalForm", "Aktiengesellschaft (AG)", "AG"),
    # -- adversarial grounding: unknowns must survive untouched --------------
    ("grounding", "country", "Atlantis", "Atlantis"),
    ("grounding", "country", "Bavaria", "Bavaria"),        # region, NOT "DE"
    ("grounding", "legalForm", "GmbbH", "GmbbH"),          # typo, NOT "GmbH"
    ("grounding", "currency", "XYZ", "XYZ"),
    ("grounding", "status", "vielleicht", "vielleicht"),
    ("grounding", "baseUnit", "barrel", "BARREL"),
    ("grounding", "recordType", "supplier", "supplier"),
    ("grounding", "validFrom", "99.99.2024", "99.99.2024"),
    ("grounding", "validFrom", "2024-13-45", "2024-13-45"),
    # -- semantic alias: embedding near-misses (NOT in the deterministic map) --
    ("semantic_alias", "country", "Nederlands", "NL"),
    ("semantic_alias", "country", "Federal Republic of Germany", "DE"),
    ("semantic_alias", "country", "French Republic", "FR"),
]


_EMBEDDINGS_ACTIVE = os.environ.get("USE_EMBEDDINGS") == "1"


def build_examples() -> list[dict]:
    rows = []
    skipped = 0
    for category, field, messy_val, expected in CASES:
        if category == "semantic_alias" and not _EMBEDDINGS_ACTIVE:
            skipped += 1
            continue  # omit, do not emit hollow identity targets
        messy = {field: messy_val}
        target, changes = normalize_record(messy)
        got = target[field]
        assert got == expected, (
            f"REGRESSION in pinned case [{category}] {field}: "
            f"{messy_val!r} -> {got!r}, expected {expected!r}. "
            f"A core/ or convention change altered documented behavior.")
        target["confidence"] = 1.0 if not rule_violations("mdm_record", target) else 0.6
        target["changes"] = changes
        rows.append({"category": category, "messages": [
            {"role": "system", "content": system_prompt("mdm_record")},
            {"role": "user", "content": json.dumps(messy, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ]})
    return rows, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/adversarial.jsonl")
    args = ap.parse_args()
    rows, skipped = build_examples()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_cat: dict[str, int] = {}
    for r in rows:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    print(f"adversarial suite: {len(rows)} pinned cases -> {args.out}")
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat:11s}: {n}")
    if skipped:
        print(f"  (skipped {skipped} semantic_alias cases; set USE_EMBEDDINGS=1 to include)")
    print("all emitted pinned expectations verified against the algorithm.")


if __name__ == "__main__":
    main()
