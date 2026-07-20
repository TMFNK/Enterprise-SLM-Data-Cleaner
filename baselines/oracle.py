#!/usr/bin/env python3
"""baselines/oracle.py: score a fixture with normalize_record (rules peer)."""
from __future__ import annotations
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))

if __name__ == "__main__":
    if "--algorithm" not in sys.argv:
        sys.argv.insert(1, "--algorithm")
    import evaluate
    evaluate.main()
