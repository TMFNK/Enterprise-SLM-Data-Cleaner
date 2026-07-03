"""
convention_spec.py: the single source of truth for a GENERIC SAP master-data
(MDM) cleaner. v1 task: normalize messy records to a documented house standard.

Design notes
------------
* Object-agnostic. Rules are keyed by FIELD TYPE (country, currency, legal form,
  IBAN, date...), not by SAP object, so the same convention covers vendor,
  customer, material, cost-center... records. `FIELD_REGISTRY` maps friendly,
  generic field names to a field type.
* Friendly generic field names (recordId, country, vatId...), not SAP table
  codes: chosen for a public repo's readability.
* Deterministic algorithm. `normalize_record()` implements the convention in code.
  For the v1 normalize task this doubles as (a) the label generator for
  synthetic data and (b) the ground-truth for eval. The fine-tuned LLM learns
  these rules so it generalizes to messiness the rules don't explicitly cover.
* Convention as data. The vocabularies, alias maps and field registry live in
  a YAML spec (conventions/default.yaml). Per-client conventions are a copied,
  edited YAML file selected via the CONVENTION env var: no code changes.
* No client data. Every value here is invented.

Pure Python, no model or network required.
"""
from __future__ import annotations
import os
import re
from datetime import datetime

import yaml

# --------------------------------------------------------------------------- #
# 1. Controlled vocabularies + alias maps, loaded from the convention spec
#    (canonical value on the right). Override with CONVENTION=path/to/spec.yaml
# --------------------------------------------------------------------------- #
_DEFAULT_SPEC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "conventions", "default.yaml")
CONVENTION_PATH = os.environ.get("CONVENTION") or _DEFAULT_SPEC

with open(CONVENTION_PATH, encoding="utf-8") as _fh:
    _SPEC = yaml.safe_load(_fh)

RECORD_TYPES = set(_SPEC["record_types"])
LEGAL_FORMS = set(_SPEC["legal_forms"])
LEGAL_FORM_ALIASES = dict(_SPEC["legal_form_aliases"])
COUNTRIES = set(_SPEC["countries"])
COUNTRY_ALIASES = dict(_SPEC["country_aliases"])
CURRENCIES = set(_SPEC["currencies"])
CURRENCY_ALIASES = dict(_SPEC["currency_aliases"])
UNITS = set(_SPEC["units"])
UNIT_ALIASES = dict(_SPEC["unit_aliases"])
STATUS = set(_SPEC["statuses"])
STATUS_ALIASES = dict(_SPEC["status_aliases"])

# Missing/unknown is encoded ONE way everywhere: JSON null. Never a sentinel.
EMPTY_STRINGS = set(_SPEC["empty_strings"])
NUMERIC_SENTINELS = set(_SPEC["numeric_sentinels"])


# --------------------------------------------------------------------------- #
# 2. Field-type normalizers (deterministic)
# --------------------------------------------------------------------------- #
def _blank(v) -> bool:
    return (v is None
            or (isinstance(v, str) and v.strip() in EMPTY_STRINGS)
            or (isinstance(v, (int, float)) and v in NUMERIC_SENTINELS))

def clean_text(v):
    if _blank(v):
        return None
    return " ".join(str(v).split())

def _lookup(v, exact: set, alias: dict):
    if _blank(v):
        return None
    s = " ".join(str(v).split())
    if s in exact:
        return s
    return alias.get(s.lower(), s)   # unknown -> returned as-is (flagged by rules)

def norm_legal_form(v): return _lookup(v, LEGAL_FORMS, LEGAL_FORM_ALIASES)
def norm_status(v):     return _lookup(v, STATUS, STATUS_ALIASES)

def norm_country(v):
    r = _lookup(v, COUNTRIES, COUNTRY_ALIASES)
    return r.upper() if isinstance(r, str) and len(r) == 2 else r

def norm_currency(v):
    if _blank(v):
        return None
    s = str(v).strip()
    return CURRENCY_ALIASES.get(s.lower(), s.upper())

def norm_unit(v):
    if _blank(v):
        return None
    s = str(v).strip()
    return UNIT_ALIASES.get(s.lower(), s.upper())

def norm_email(v):
    if _blank(v):
        return None
    return str(v).strip().lower()

def norm_iban(v):
    if _blank(v):
        return None
    return re.sub(r"\s+", "", str(v)).upper()

def norm_vat(v):
    if _blank(v):
        return None
    return re.sub(r"[\s.-]", "", str(v)).upper()

def norm_phone(v):
    """Simplified E.164: keep digits, map leading 00 -> +, require + prefix."""
    if _blank(v):
        return None
    s = str(v).strip()
    plus = s.startswith("+") or s.startswith("00")
    digits = re.sub(r"\D", "", s)
    if s.startswith("00"):
        digits = digits[2:]
    return ("+" + digits) if plus and digits else (digits or None)

def norm_date(v):
    """Parse common formats -> ISO-8601 (YYYY-MM-DD). German DD.MM assumed."""
    if _blank(v):
        return None
    s = str(v).strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return s   # unparseable -> as-is (flagged by rules)

def norm_amount(v):
    """Normalize decimal: handle '1.234,56' (de) and '1,234.56' (en) -> float."""
    if _blank(v):
        return None
    s = re.sub(r"[^\d,.\-]", "", str(v))
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") \
            else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        f = float(s)
    except ValueError:
        return None
    return None if f in NUMERIC_SENTINELS else f

def norm_record_type(v):
    s = clean_text(v)
    return _RECORD_TYPE_BY_LOWER.get(s.lower(), s) if s else None

_RECORD_TYPE_BY_LOWER = {t.lower(): t for t in RECORD_TYPES}

def norm_id(v):
    return None if _blank(v) else str(v).strip().upper()


# --------------------------------------------------------------------------- #
# 3. Field registry: friendly field name -> normalizer, driven by the spec's
#    `fields:` map (field name -> field type -> normalizer function)
# --------------------------------------------------------------------------- #
_NORMALIZERS_BY_TYPE = {
    "id": norm_id,
    "recordType": norm_record_type,
    "text": clean_text,
    "legalForm": norm_legal_form,
    "country": norm_country,
    "vat": norm_vat,
    "iban": norm_iban,
    "email": norm_email,
    "phone": norm_phone,
    "currency": norm_currency,
    "unit": norm_unit,
    "status": norm_status,
    "date": norm_date,
    "amount": norm_amount,
}

FIELD_REGISTRY = {name: _NORMALIZERS_BY_TYPE[ftype]
                  for name, ftype in _SPEC["fields"].items()}


def normalize_record(record: dict) -> tuple[dict, list[str]]:
    """The deterministic ALGORITHM: clean a whole record, return (clean, changes)."""
    out, changes = {}, []
    for key, val in record.items():
        norm = FIELD_REGISTRY.get(key)
        new = norm(val) if norm else clean_text(val)
        out[key] = new
        if new != val:
            changes.append(f"{key}: {val!r} -> {new!r}")
    return out, changes


# --------------------------------------------------------------------------- #
# 4. Block schema (one MDM record = one block) + rule checks
# --------------------------------------------------------------------------- #
_STR = {"type": ["string", "null"]}
BLOCK_SCHEMAS = {
    "mdm_record": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            **{k: _STR for k in FIELD_REGISTRY},
            "recordType": {"type": ["string", "null"], "enum": sorted(RECORD_TYPES) + [None]},
            "country": {"type": ["string", "null"], "enum": sorted(COUNTRIES) + [None]},
            "currency": {"type": ["string", "null"], "enum": sorted(CURRENCIES) + [None]},
            "baseUnit": {"type": ["string", "null"], "enum": sorted(UNITS) + [None]},
            "status": {"type": ["string", "null"], "enum": sorted(STATUS) + [None]},
            "legalForm": {"type": ["string", "null"], "enum": sorted(LEGAL_FORMS) + [None]},
            "amount": {"type": ["number", "null"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "changes": {"type": "array", "items": {"type": "string"}},
        },
    },
}

# Field types whose cleaning is a reasoning/judgement problem (future v2/v3:
# dedup, golden-record merge, nested-JSON reshape). Not triggered by v1.
REASONING_BLOCKS = {"dedup", "reshape"}

_CONTROLLED = {"country": COUNTRIES, "currency": CURRENCIES, "baseUnit": UNITS,
               "status": STATUS, "legalForm": LEGAL_FORMS, "recordType": RECORD_TYPES}


def rule_violations(block_type: str, obj: dict) -> list[str]:
    """Semantic gates the JSON grammar can't enforce ([] == clean)."""
    v: list[str] = []
    if block_type != "mdm_record":
        return v
    for field, allowed in _CONTROLLED.items():
        val = obj.get(field)
        if val is not None and val not in allowed:
            v.append(f"{field} '{val}' not in controlled list")
    for field in ("name1", "name2", "street", "city"):
        val = obj.get(field)
        if isinstance(val, str) and val != " ".join(val.split()):
            v.append(f"{field} has stray whitespace")
    if isinstance(obj.get("email"), str) and obj["email"] != obj["email"].lower():
        v.append("email not lower-cased")
    if isinstance(obj.get("iban"), str) and (" " in obj["iban"] or not obj["iban"].isupper()):
        v.append("iban has spaces or is not upper-case")
    for field, val in obj.items():
        if isinstance(val, str) and val.strip() in EMPTY_STRINGS:
            v.append(f"{field} should be null, not {val!r}")
        if isinstance(val, (int, float)) and val in NUMERIC_SENTINELS:
            v.append(f"{field} holds sentinel {val}; should be null")
    return v


def system_prompt(block_type: str) -> str:
    return (
        "You normalize a messy master-data record to the house convention. Rules: "
        "trim and collapse whitespace; encode any missing/unknown value as JSON null "
        "(never '-', 'n/a', -1, etc.); map country to ISO-2, currency to ISO-4217, "
        "unit of measure and legal form and status to their controlled code; lower-case "
        "email; upper-case and de-space IBAN/VAT; dates to ISO-8601 (YYYY-MM-DD); amounts "
        "to a plain decimal number. Return ONLY the cleaned record as JSON, plus a 0-1 "
        "`confidence` and a `changes` list naming each edit. Never invent data."
    )


if __name__ == "__main__":
    messy = {
        "recordId": "v-1001", "recordType": "vendor", "name1": "  Muster  Handels  ",
        "legalForm": "Gesellschaft mit beschränkter Haftung", "city": "München ",
        "country": "Germany", "vatId": "de 811 569 869", "iban": "de89 3704 0044 0532 0130 00",
        "email": "INFO@Muster.DE", "currency": "€", "baseUnit": "pcs",
        "status": "aktiv", "validFrom": "01.03.2024", "amount": "1.234,56", "phone": "0049 (30) 12345",
    }
    clean, changes = normalize_record(messy)
    assert clean["country"] == "DE" and clean["currency"] == "EUR"
    assert clean["baseUnit"] == "PCE" and clean["status"] == "active"
    assert clean["legalForm"] == "GmbH" and clean["email"] == "info@muster.de"
    assert clean["iban"] == "DE89370400440532013000" and clean["amount"] == 1234.56
    assert clean["validFrom"] == "2024-03-01" and clean["name1"] == "Muster Handels"
    assert rule_violations("mdm_record", clean) == []
    assert rule_violations("mdm_record", messy)   # messy input has violations
    # numeric sentinels -> null, recordType casing -> controlled value
    edge, _ = normalize_record({"recordType": "Vendor", "amount": -1, "validTo": -999})
    assert edge == {"recordType": "vendor", "amount": None, "validTo": None}
    assert normalize_record({"amount": "-999"})[0]["amount"] is None
    assert rule_violations("mdm_record", edge) == []
    print("convention_spec self-check passed.")
    print(f"  {len(FIELD_REGISTRY)} fields, {len(changes)} edits on the demo record.")
    for c in changes:
        print("   -", c)
