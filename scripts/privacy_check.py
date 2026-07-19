#!/usr/bin/env python3
"""Privacy gate for tracked paths.

Fails if:
  - Any data file is committed under fixtures/real/ outside local/ (real extracts).
  - IBAN-like or DE-VAT-like tokens appear under fixtures/real/ (non-local).

Synthetic gold under fixtures/gold.jsonl may contain invented IBAN/VAT shapes;
those are allowed. Put real extracts only in fixtures/real/local/ (gitignored).
"""
from __future__ import annotations
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = os.path.join(ROOT, "fixtures", "real")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
DE_VAT_RE = re.compile(r"\bDE\s?\d{9}\b", re.IGNORECASE)
DATA_SUFFIXES = (".jsonl", ".json", ".csv", ".tsv", ".xlsx", ".txt")


def main() -> int:
    errors: list[str] = []
    if os.path.isdir(REAL):
        for dirpath, dirnames, files in os.walk(REAL):
            # do not descend into local/ (gitignored extracts)
            dirnames[:] = [d for d in dirnames if d != "local"]
            rel = os.path.relpath(dirpath, REAL)
            for name in files:
                if name in ("README.md", ".gitkeep"):
                    continue
                path = os.path.join(dirpath, name)
                if name.endswith(DATA_SUFFIXES) or rel != ".":
                    errors.append(
                        f"{path}: real-extract file must live under "
                        f"fixtures/real/local/ (gitignored), not in git"
                    )
                    continue
                try:
                    text = open(path, encoding="utf-8").read()
                except (UnicodeDecodeError, OSError):
                    continue
                for m in IBAN_RE.finditer(text):
                    errors.append(f"{path}: IBAN-like {m.group(0)}")
                for m in DE_VAT_RE.finditer(text):
                    errors.append(f"{path}: DE-VAT-like {m.group(0)}")

    if errors:
        print("PRIVACY CHECK FAILED:")
        for e in errors[:50]:
            print(f"  {e}")
        print("See docs/PRIVACY.md.")
        return 1
    print("privacy-check: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
