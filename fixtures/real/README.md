# Real-data smoke ritual

Before claiming a new fine-tune is production-ready:

1. Score oracle + (optional) base model on `fixtures/gold.jsonl`.
2. Place **~20 anonymized** real MDM lines in `local/` (gitignored).
3. Run them through `runtime/clean.py` (algorithm and/or `--live`).
4. Note failure modes (legal-form variants, VAT shapes, junk keys) in a
   short local note or a redacted summary under `reports/`.

```bash
# example: algorithm-only smoke on a local file
python3 runtime/clean.py --batch fixtures/real/local/smoke.jsonl \
  --out /tmp/smoke-out.jsonl --audit-dir audit
```

Never commit files from `local/`. See [docs/PRIVACY.md](../../docs/PRIVACY.md).
