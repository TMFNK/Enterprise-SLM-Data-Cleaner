# Training with Apple MLX

Fine-tune locally on Apple Silicon, then export to GGUF for llama.cpp. The Makefile
sets `MODEL`, `ALIAS`, and `GGUF_QUANT` from `MODEL_PRESET` (default:
`qwen3-0.6b`). Run `make list-models` for options.

## 0. Install

```bash
pip install mlx-lm
# llama.cpp for conversion + serving:  brew install llama.cpp
```

## 1. Data

`synth/generate.py` writes `data/train.jsonl`, `data/valid.jsonl`, `data/test.jsonl`
in the chat format MLX expects (`{"messages": [...]}`).

```bash
python synth/generate.py --n 2000 --out data --seed 0
```

Start small and plot a learning curve (250 → 500 → 1k → 2k): data is free, so
train on increasing slices and stop where eval flattens.

## 2. LoRA fine-tune

```bash
make train
# or manually:
mlx_lm.lora \
  --model "$MODEL" \
  --train \
  --data ./data \
  --iters 1000 \
  --batch-size 4 \
  --num-layers 8 \
  --adapter-path adapters
```

Tip: run the model with thinking disabled for this task (terse JSON, not
chain-of-thought): pass `/no_think` in the system prompt if you keep Qwen3's
thinking template.

## 3. Fuse + convert to GGUF

```bash
make fuse gguf
# or manually (replace ALIAS / GGUF_QUANT with your MODEL_PRESET values):
mlx_lm.fuse --model "$MODEL" --adapter-path adapters --save-path fused
python /path/to/llama.cpp/convert_hf_to_gguf.py fused --outfile "${ALIAS}.gguf"
llama-quantize "${ALIAS}.gguf" "${ALIAS}-q8_0.gguf" Q8_0
```

(For a 0.6B model use a high-bit quant: Q8_0 / Q6_K: the RAM cost is tiny and
JSON fidelity is better.)

## 4. Serve

```bash
make serve
# or manually:
llama-server -m "${ALIAS}-q8_0.gguf" --port 8080 --alias "${ALIAS}"
```

## 5. Evaluate the before/after

```bash
# baseline: stock instruct model on :8080 (no adapter): establishes zero-shot
python eval/evaluate.py --data data/test.jsonl --live --model-name "${ALIAS}"
# then swap in the fine-tuned GGUF at the same port and re-run to see the lift
```
