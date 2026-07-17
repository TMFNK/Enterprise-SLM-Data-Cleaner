"""
synth/generate.py: synthetic (messy -> clean) MDM training data.

Pipeline per example:
    1. build a CLEAN, convention-valid record (invented values)
    2. CORRUPT it into a realistic messy record
    3. label it with the deterministic algorithm:  target = normalize_record(messy)
    4. emit a chat example {system, user=messy, assistant=target}

Because the algorithm produces the label, every pair is correct by construction and
the dataset is self-filtering (no teacher LLM needed). All data is invented: no
client data of any kind.

Usage:
    python synth/generate.py --n 1000 --out data --seed 0
"""
from __future__ import annotations
import os
import sys
import json
import random
import argparse

# make convention_spec (in core/) importable when run from anywhere
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
import convention_spec as spec
from convention_spec import normalize_record, rule_violations, system_prompt

# --------------------------------------------------------------------------- #
# Invented value pools
# --------------------------------------------------------------------------- #
STEMS = ["Muster", "Nordwind", "Rheintal", "Alpenblick", "Hansa", "Bergmann",
         "Vogel", "Kruger", "Weissbach", "Baumann", "Lindt", "Falken", "Meridian",
         "Adler", "Sonne", "Kranich", "Eiche", "Delphin", "Orion", "Tannenhof"]
STREETS = ["Hauptstrasse", "Bahnhofstrasse", "Lindenweg", "Gartenstrasse",
           "Industriestrasse", "Am Markt", "Ringstrasse", "Feldweg", "Schulstrasse"]
CITY_COUNTRY = [("Muenchen", "DE"), ("Hamburg", "DE"), ("Berlin", "DE"),
                ("Wien", "AT"), ("Zuerich", "CH"), ("Paris", "FR"), ("Milano", "IT"),
                ("Amsterdam", "NL"), ("Warschau", "PL"), ("London", "GB")]
CCY_BY_COUNTRY = {"DE": "EUR", "AT": "EUR", "FR": "EUR", "IT": "EUR", "NL": "EUR",
                  "CH": "CHF", "PL": "PLN", "GB": "GBP"}
PHONE_CC = {"DE": "49", "AT": "43", "CH": "41", "FR": "33", "IT": "39",
            "NL": "31", "PL": "48", "GB": "44"}
MATERIALS = ["Edelstahlschraube M6", "Zitronensaftkonzentrat", "Kartonverpackung",
             "Klebeband transparent", "Holzpalette EUR", "Etikett Rolle",
             "Schmiermittel HD", "Dichtungsring 20mm"]
ID_PREFIX = {"vendor": "V", "customer": "C", "material": "M",
             "costCenter": "K", "glAccount": "G"}


def _invert(alias_map: dict) -> dict:
    """canonical -> [messy variants that normalize to it]."""
    out: dict[str, list[str]] = {}
    for messy, canon in alias_map.items():
        out.setdefault(canon, []).append(messy)
    return out


COUNTRY_VARIANTS = _invert(spec.COUNTRY_ALIASES)
CURRENCY_VARIANTS = _invert(spec.CURRENCY_ALIASES)
LEGAL_VARIANTS = _invert(spec.LEGAL_FORM_ALIASES)
UNIT_VARIANTS = _invert(spec.UNIT_ALIASES)
STATUS_VARIANTS = _invert(spec.STATUS_ALIASES)
MISSING_TOKENS = ["", " ", "-", "n/a", "N/A", "null", "?"]

# Near-miss variants NOT listed in the alias map — only the embedding layer
# can reverse these. Applied only when USE_EMBEDDINGS=1 so labels stay correct.
_EMBEDDINGS_ACTIVE = os.environ.get("USE_EMBEDDINGS") == "1"
EMBEDDING_NEAR_MISSES = {
    "country": [
        ("Nederlands", "NL"),
        ("Federal Republic of Germany", "DE"),
        ("French Republic", "FR"),
    ],
}


# --------------------------------------------------------------------------- #
# 1. clean record
# --------------------------------------------------------------------------- #
def clean_record(rtype: str, rng: random.Random) -> dict:
    rid = f"{ID_PREFIX[rtype]}{rng.randint(10000, 99999)}"
    status = rng.choices(["active", "inactive", "blocked"], weights=[8, 1, 1])[0]
    year = rng.randint(2016, 2025)
    valid_from = f"{year:04d}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
    rec: dict = {"recordId": rid, "recordType": rtype,
                 "status": status, "validFrom": valid_from}

    if rtype in ("vendor", "customer"):
        stem = rng.choice(STEMS)
        lf = rng.choice(sorted(spec.LEGAL_FORMS))
        city, country = rng.choice(CITY_COUNTRY)
        rec.update({
            "name1": f"{stem} {rng.choice(['Handels', 'Industrie', 'Technik', 'Food', 'Logistik'])}",
            "legalForm": lf,
            "street": rng.choice(STREETS), "houseNo": str(rng.randint(1, 199)),
            "postalCode": f"{rng.randint(10000, 99999)}", "city": city, "country": country,
            "vatId": f"{country}{rng.randint(100000000, 999999999)}",
            "iban": f"{country}{rng.randint(10, 99)}{rng.randint(10**16, 10**17 - 1)}",
            "email": f"info@{stem.lower()}.example",
            "phone": f"+{PHONE_CC.get(country, '49')}{rng.randint(1000000, 9999999)}",
            "currency": CCY_BY_COUNTRY.get(country, "EUR"),
        })
    elif rtype == "material":
        _, country = rng.choice(CITY_COUNTRY)
        rec.update({
            "name1": rng.choice(MATERIALS),
            "baseUnit": rng.choice(sorted(spec.UNITS)),
            "amount": round(rng.uniform(0.5, 950.0), 2),
            "currency": CCY_BY_COUNTRY.get(country, "EUR"),
        })
    else:  # costCenter / glAccount
        rec.update({
            "name1": f"{rng.choice(['Vertrieb', 'Einkauf', 'Produktion', 'IT', 'Logistik'])} {rng.randint(1, 9)}",
            "currency": "EUR",
        })
    return rec


# --------------------------------------------------------------------------- #
# 2. corruptor
# --------------------------------------------------------------------------- #
def _messy_case(rng, s):
    r = rng.random()
    if r < 0.33:
        return s.upper()
    if r < 0.66:
        return s.lower()
    return s


def corrupt_record(clean: dict, rng: random.Random) -> dict:
    m = dict(clean)

    def maybe(p):
        return rng.random() < p

    # whitespace / case on free-text
    for f in ("name1", "name2", "street", "city"):
        if f in m and isinstance(m[f], str) and maybe(0.5):
            pad = " " * rng.randint(1, 3)
            m[f] = rng.choice([pad + m[f], m[f] + pad, m[f].replace(" ", "  ", 1)])
    # controlled-vocab -> a messy alias variant
    if "country" in m and COUNTRY_VARIANTS.get(m["country"]) and maybe(0.7):
        m["country"] = _messy_case(rng, rng.choice(COUNTRY_VARIANTS[m["country"]]))
    if "currency" in m and CURRENCY_VARIANTS.get(m["currency"]) and maybe(0.7):
        m["currency"] = rng.choice(CURRENCY_VARIANTS[m["currency"]])
    if "legalForm" in m and LEGAL_VARIANTS.get(m["legalForm"]) and maybe(0.6):
        m["legalForm"] = _messy_case(rng, rng.choice(LEGAL_VARIANTS[m["legalForm"]]))
    if "baseUnit" in m and UNIT_VARIANTS.get(m["baseUnit"]) and maybe(0.7):
        m["baseUnit"] = rng.choice(UNIT_VARIANTS[m["baseUnit"]])
    if "status" in m and STATUS_VARIANTS.get(m["status"]) and maybe(0.7):
        m["status"] = _messy_case(rng, rng.choice(STATUS_VARIANTS[m["status"]]))
    # embedding near-miss corruption — only when embeddings can reverse it
    if _EMBEDDINGS_ACTIVE and "country" in m and maybe(0.08):
        for messy, canon in EMBEDDING_NEAR_MISSES["country"]:
            if m["country"] == canon:
                m["country"] = messy
                break
    # format munging
    if "iban" in m and maybe(0.7):
        v = m["iban"]
        m["iban"] = " ".join(v[i:i + 4] for i in range(0, len(v), 4)).lower()
    if "vatId" in m and maybe(0.6):
        v = m["vatId"]
        m["vatId"] = (v[:2] + " " + " ".join(v[2:][i:i + 3] for i in range(0, len(v) - 2, 3))).lower()
    if "email" in m and maybe(0.5):
        m["email"] = _messy_case(rng, m["email"])
    if "phone" in m and maybe(0.6):
        digits = m["phone"].lstrip("+")
        m["phone"] = rng.choice([f"00{digits}", f"+{digits[:2]} ({digits[2:5]}) {digits[5:]}"])
    if "validFrom" in m and maybe(0.7):
        y, mo, d = m["validFrom"].split("-")
        m["validFrom"] = rng.choice([f"{d}.{mo}.{y}", f"{d}/{mo}/{y}"])
    if "amount" in m and maybe(0.7):
        de = f"{m['amount']:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        m["amount"] = rng.choice([de, f"EUR {de}", f"{m['amount']}"])
    if "recordId" in m and maybe(0.4):
        m["recordId"] = _messy_case(rng, m["recordId"])
    # inject a missing/sentinel value on an optional field
    if maybe(0.3):
        opt = [f for f in ("name2", "region", "taxNumber", "bic") if f in m] or ["name2"]
        m[rng.choice(opt)] = rng.choice(MISSING_TOKENS)
    return m


# --------------------------------------------------------------------------- #
# 3. example builder + writer
# --------------------------------------------------------------------------- #
def make_example(rng: random.Random, rtypes, weights):
    rtype = rng.choices(rtypes, weights=weights)[0]
    clean = clean_record(rtype, rng)
    assert rule_violations("mdm_record", clean) == [], f"dirty clean: {clean}"
    messy = corrupt_record(clean, rng)
    target, changes = normalize_record(messy)
    target["confidence"] = 1.0 if not rule_violations("mdm_record", target) else 0.6
    target["changes"] = changes
    return {"messages": [
        {"role": "system", "content": system_prompt("mdm_record")},
        {"role": "user", "content": json.dumps(messy, ensure_ascii=False)},
        {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
    ]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="total examples")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--valid-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rtypes = ["vendor", "customer", "material", "costCenter", "glAccount"]
    weights = [4, 3, 3, 1, 1]
    rows = [make_example(rng, rtypes, weights) for _ in range(args.n)]
    rng.shuffle(rows)

    n_test = int(args.n * args.test_frac)
    n_valid = int(args.n * args.valid_frac)
    splits = {"test": rows[:n_test],
              "valid": rows[n_test:n_test + n_valid],
              "train": rows[n_test + n_valid:]}

    os.makedirs(args.out, exist_ok=True)
    for name, data in splits.items():
        path = os.path.join(args.out, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for r in data:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {name:5s}: {len(data):5d} -> {path}")
    print(f"done. seed={args.seed}, total={args.n}")


if __name__ == "__main__":
    main()
