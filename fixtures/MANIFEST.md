# Fixtures manifest

- convention_path: `conventions/default.yaml`
- convention_sha256: `5e59eb4e41b612c9c73bb44e6ebf31cbb793e657df25e8f4c3b9deb582b02cdc`
- gold.jsonl: 100 rows, seed=42
- holdout_unseen_noise.jsonl: 40 rows, seed=99
- gold is sacred: never overwritten by `synth/generate.py`
- unseen corruption family is excluded from training corruptors
