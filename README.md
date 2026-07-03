# Enterprise SLM Data Cleaner

Enterprise-grade version of [Local-SLM-Data-Cleaner](https://github.com/TMFNK/Local-SLM-Data-Cleaner):
a small language model (SLM), fine-tuned entirely on synthetic data, that
normalizes unclean SAP-style master data to a documented house convention. It is built
to run in secure, air-gapped enterprise environments.

Where the original repo is a beginner-friendly demo for a Mac laptop, this
version targets production deployments: editable client-specific convention specs,
a (manual) review queue with an append-only audit trail, containerized offline
serving with vendored model weights, and eval-gated releases.

**Status: early extraction phase.** The proven core (deterministic convention
algorithm, synthetic data generator, eval harness, model-with-safety-net
runtime) has been carried over from the origin repo. The enterprise layers are
being built on top, in this order:

1. **Convention-as-Spec**: client specific YAML/markdown convention files that
   drive the validator and the data generator (no code changes per client)
2. **Audit log + review queue**: append-only per-record log (input, output,
   changes, confidence, model/adapter hash, spec version, timestamp);
   low-confidence records go to manual review
3. **Deployment hardening**: container with no network egress, vendored
   weights with pinned hashes, zero runtime downloads
4. **Eval gating**: CI blocks merges if field accuracy drops against the
   adversarial eval set

## Layout

```bash
core/       the house convention + deterministic algorithm (single source of truth)
synth/      synthetic messy->clean data generator (no real data, ever)
eval/       eval harness: valid-JSON rate, exact-record match, field accuracy
runtime/    clean service: model -> validate -> algorithm safety net
train/      MLX LoRA fine-tuning notes
Makefile    every pipeline step as `make <command>` (see `make help`)
```

## Quick start (pipeline)

```bash
make setup        # install Python deps + mlx-lm
make data         # generate synthetic train/valid/test data
make sanity       # verify the data against the rule-based algorithm (~100%)
make train        # LoRA fine-tune Qwen3-0.6B (Apple Silicon / MLX)
```

For the full step-by-step walkthrough and the concepts behind the approach
(knowledge distillation, LoRA, quantization, grammar-constrained decoding),
see the [origin repo's README](https://github.com/TMFNK/Local-SLM-Data-Cleaner).

## Credit

Built on [Local-SLM-Data-Cleaner](https://github.com/TMFNK/Local-SLM-Data-Cleaner)
by [mbitai](https://www.mbitai.com), which remains the public demo and
tutorial for this approach. All sample data in both repos is synthetic and
invented; no client data is involved at any point.

## License

AGPL-3.0 (see [LICENSE](LICENSE)).

For commercial licensing without AGPL obligations, or help applying this to
your own master data migration or data-quality work, contact
[mbitai.com](https://www.mbitai.com).
