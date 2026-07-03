# Deployment: air-gapped container

Build in a trusted, connected environment. Run with **no network at all**.
The security posture, in one sentence: the image carries everything it needs
(server, runtime, hash-pinned weights), so the running container can be denied
network access entirely and still do its job.

## 1. Vendor the weights (once, after training)

```bash
make gguf         # produces the quantized GGUF (see the main README)
make pin-model    # copies it to models/ and pins its sha256 in models/MANIFEST.sha256
```

Commit `models/MANIFEST.sha256` (the weights themselves are gitignored; ship
them alongside the repo, e.g. in your artifact registry). Anyone can check the
weights they received match what you pinned:

```bash
make verify-model
```

The container entrypoint runs the same check and **refuses to start** if the
weights don't match the manifest.

## 2. Build

```bash
docker build -f deploy/Containerfile -t enterprise-slm-cleaner .
```

## 3. Run

**Batch cleaning, fully air-gapped** — the showcase mode. `--network none`
removes the container's network stack entirely; records move only via the
mounted volume, and the audit log + review queue land there too:

```bash
mkdir -p exchange && cp your-records.jsonl exchange/in.jsonl
docker run --rm --network none \
  -v "$PWD/exchange:/data" \
  enterprise-slm-cleaner batch /data/in.jsonl /data/out.jsonl
# results: exchange/out.jsonl, audit trail: exchange/audit/
```

**Long-running server** — for pod-internal use where *you* define the network
(e.g. a compose/K8s network shared only with your integration service):

```bash
docker run --rm -p 127.0.0.1:8080:8080 enterprise-slm-cleaner serve
```

## 4. What a security review can verify

- `--network none`: no egress, no ingress, no DNS — enforced by the runtime,
  not by promises in application code.
- `models/MANIFEST.sha256` is in git history; the entrypoint fails closed on
  any weight mismatch (supply-chain check at every start).
- No credentials, tokens or client data in the image: only code, convention
  specs and synthetic-data-trained weights.
- Every cleaning decision is in the append-only audit log under `/data/audit`
  (input, output, violations, confidence, model hash, convention hash).

## Status / honesty note

The `pin-model` / `verify-model` targets and the entrypoint's manifest check
are tested. The container image itself has not yet been built in this repo's
CI or on a dev machine without Docker — treat the first build as a smoke test
(the base image's `llama-server` path is expected at `/app/llama-server`;
override with `LLAMA_BIN` if your base image differs).
