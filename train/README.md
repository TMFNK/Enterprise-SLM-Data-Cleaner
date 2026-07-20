# Training (Apple MLX)

Pipeline steps live in the root README and Makefile:

```bash
make list-models
make data check-balance train fuse gguf serve eval
```

Tips not covered there:

- Plot a learning curve (250 → 500 → 1k → 2k); score **pinned gold**
  (`fixtures/gold.jsonl`) for before/after, not only `data/test.jsonl`.
- For Qwen3, disable thinking for this task (`/no_think` in the system prompt)
  so outputs stay terse JSON.
- Prefer a high-bit quant (Q8_0 / Q6_K) for 0.6B: RAM cost is tiny, JSON
  fidelity is better.
