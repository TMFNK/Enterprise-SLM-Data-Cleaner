# Privacy defaults

This project is built so **client master data never needs to leave the house**
and **training never uses real records**. Keep it that way in fixtures and
docs that land in git.

## Rules

1. **Training data is synthetic.** `synth/generate.py` invents names, IBANs,
   and VAT-shaped strings. Do not feed client extracts into `make data`.
2. **Pinned gold may use invented identifiers.** `fixtures/gold.jsonl` is
   synthetic; invented IBAN/VAT _shapes_ are OK.
3. **Real batches stay local.** Put them in `fixtures/real/local/` (gitignored).
   Summaries of failure modes may be committed; the rows themselves must not.
4. **Air-gap delivery.** Production serving uses `deploy/` with `--network none`
   and pinned model fingerprints.

## Commands

```bash
make privacy-check
```

Fails if any data file is committed under `fixtures/real/` outside `local/`.

## Confidence and review

Model output below `--min-confidence` (default `0.9` in `runtime/clean.py`)
goes to `audit/review-queue.jsonl`. Never treat low confidence as silent
auto-accept.
