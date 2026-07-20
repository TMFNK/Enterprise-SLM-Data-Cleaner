# Convention specs (product surface)

This folder is what a **data steward** edits. No Python changes are required
to customize house standards for a client.

## Workflow

1. Copy `default.yaml` → `conventions/<client>.yaml` (or edit `default.yaml` for the demo).
2. Change vocabularies, aliases, empty tokens, or `embedding_thresholds` as needed.
3. Point the pipeline at the file:

   ```bash
   make data CONVENTION=conventions/<client>.yaml
   make sanity CONVENTION=conventions/<client>.yaml
   make eval-gate CONVENTION=conventions/<client>.yaml
   ```

4. Re-run eval and (if you train) regenerate data **before** LoRA. The audit
   log records the convention path + SHA-256 of the YAML that was in force.

## What lives here vs in code

| In YAML (edit freely) | In `core/convention_spec.py` (code) |
|---|---|
| Controlled sets (countries, legal forms, …) | Parsers (dates, amounts, IBAN, phone) |
| Alias maps (messy → canonical) | Field-type dispatch (`FIELD_REGISTRY`) |
| Empty / sentinel encodings | Schema + `normalize_record()` |
| Embedding thresholds | Embedding resolver wiring |

## Rules of the road

- New client = new YAML file, not a fork of the repo.
- Keep values invented / synthetic in the public demo files.
- After every YAML edit: `make privacy-check` and `make eval-gate`.
- Do not put real IBANs, USt-IdNr, or personal names in tracked YAML.

## Related

- Privacy: [docs/PRIVACY.md](../docs/PRIVACY.md)
- Eval peers (oracle · base · fine-tuned): [reports/template-eval.md](../reports/template-eval.md)
- Real-data smoke ritual: [fixtures/real/README.md](../fixtures/real/README.md)
