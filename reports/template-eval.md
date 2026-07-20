# Eval report template

Fill one row per system on the **same** fixtures. Provenance belongs in every
saved report.

## Provenance

- date:
- convention_path:
- convention_sha256:
- fixture: `fixtures/gold.jsonl` / `data/adversarial.jsonl` / `fixtures/holdout_unseen_noise.jsonl`
- model fingerprint (GGUF sha256):
- seed:

## Comparison table

| System | Validity @clean | Field accuracy @clean | Field accuracy @adversarial | Field accuracy @unseen |
|---|---|---|---|---|
| oracle (`normalize_record`) | | | | |
| base SLM | | | | |
| fine-tuned SLM | | | | |

## Ops

| Metric | Value |
|---|---|
| Review-queue rate (conf &lt; threshold) | |
| `check_balance.py` | pass / fail |
| `privacy-check` | pass / fail |

## Notes

-

## Commands

```bash
make report-oracle          # oracle on gold → reports/
make eval-gate              # CI: gold + synth + adversarial + unseen @ 100%
make oracle ORACLE_DATA=fixtures/holdout_unseen_noise.jsonl
make privacy-check
make check-balance          # after make data
```
