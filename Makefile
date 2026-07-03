# =============================================================================
# Enterprise SLM Data Cleaner: training + eval pipeline.
#
# New here? Run `make help` to list every command, then follow the numbered
# steps in the README. Each command below prints what it is doing.
#
# You override any setting on the command line, for example:
#     make data N=2000          # make 2000 examples instead of 1000
#     make train ITERS=1500     # train for more steps
# =============================================================================

# ---- settings you can override on the command line -----------------------
# Keep values on their own lines with no inline comments. A trailing-space
# comment would become part of the value and break commands like `-hf repo:quant`.
#
#   MODEL       base model to fine-tune (auto-downloads)
#   GGUF_HF     stock GGUF repo for the zero-shot baseline
#   GGUF_QUANT  quant level to download/build (Q8_0, Q6_K, ...)
#   N           number of synthetic examples
#   SEED        random seed (same seed = same data)
#   ITERS       training steps
#   BATCH       training batch size
#   LAYERS      how many layers LoRA touches
#   PORT        local server port
#   ALIAS       model name the eval/clean scripts look for
#   LLAMA_CPP   path to a llama.cpp source checkout (only for `make gguf`)
#   CONVENTION  convention spec file (per-client: conventions/<client>.yaml)
MODEL      ?= Qwen/Qwen3-0.6B
GGUF_HF    ?= Qwen/Qwen3-0.6B-GGUF
GGUF_QUANT ?= Q8_0
N          ?= 1000
SEED       ?= 0
ITERS      ?= 1000
BATCH      ?= 4
LAYERS     ?= 8
PORT       ?= 8080
ALIAS      ?= qwen3-0.6b-cleaner
DATA       ?= data
ADAPTERS   ?= adapters
FUSED      ?= fused
GGUF       ?= $(ALIAS).gguf
QGGUF      ?= $(ALIAS)-q8_0.gguf
LLAMA_CPP  ?= ../llama.cpp
PY         ?= python3
CONVENTION ?= conventions/default.yaml
export CONVENTION

.DEFAULT_GOAL := help
.PHONY: help setup model data sanity baseline-serve baseline train fuse gguf \
        serve eval demo review all clean distclean

help:  ## show this list of commands
	@echo "Enterprise SLM Data Cleaner: commands (run them in this order):"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --- STEP 3: install the tools --------------------------------------------- #
setup:  ## STEP 3: install Python libraries + the trainer (mlx-lm)
	@echo ">> Installing Python packages (requests)..."
	$(PY) -m pip install -r requirements.txt
	@echo ">> Installing mlx-lm (the fine-tuning engine for Apple Silicon)..."
	$(PY) -m pip install mlx-lm
	@echo ">> Installing llama.cpp (runs the model) via Homebrew, if available..."
	@command -v brew >/dev/null 2>&1 && brew install llama.cpp \
	  || echo "   !! Homebrew not found. Install it from https://brew.sh then run: brew install llama.cpp"
	@echo ">> Done. Next: make model"

# --- STEP 4: download the model FRESH -------------------------------------- #
model:  ## STEP 4: download the base model (Qwen3-0.6B) from Hugging Face
	@echo ">> Downloading $(MODEL) (~1.2 GB, first time only, no login needed)..."
	@echo ">> It caches in ~/.cache/huggingface so later steps are instant."
	$(PY) -c "from mlx_lm import load; load('$(MODEL)'); print('model ready')"
	@echo ">> Done. Next: make data"

# --- STEP 5: make the training data ---------------------------------------- #
data:  ## STEP 5a: generate synthetic train/valid/test data (N, SEED)
	@echo ">> Generating $(N) synthetic messy->clean examples..."
	$(PY) synth/generate.py --n $(N) --out $(DATA) --seed $(SEED)
	@echo ">> Done. Next: make sanity"

sanity:  ## STEP 5b: check the data is correct (should say 100%)
	@echo ">> Checking the held-out test split against the rule-based algorithm..."
	$(PY) eval/evaluate.py --data $(DATA)/test.jsonl --algorithm
	@echo ">> If the numbers are ~100%, the data is good. Next: make baseline-serve"

# --- STEP 6: measure the model BEFORE training (the 'before' number) ------- #
baseline-serve:  ## STEP 6a: serve the STOCK model (downloads it fresh), keep running
	@echo ">> Downloading + serving the untrained stock model on port $(PORT)."
	@echo ">> Leave this running and open a SECOND terminal for 'make baseline'."
	llama-server -hf $(GGUF_HF):$(GGUF_QUANT) --port $(PORT) --alias $(ALIAS)

baseline:  ## STEP 6b: score the stock model (run in the 2nd terminal)
	@echo ">> Scoring the untrained model (this is your 'before' score)..."
	$(PY) eval/evaluate.py --data $(DATA)/test.jsonl --live --port $(PORT)

# --- STEP 7: fine-tune ------------------------------------------------------ #
train:  ## STEP 7: fine-tune the model on your data (takes a while)
	@echo ">> Fine-tuning $(MODEL) with LoRA for $(ITERS) steps..."
	@echo ">> Stop the baseline-serve terminal first to free memory."
	mlx_lm.lora --model $(MODEL) --train --data ./$(DATA) \
	  --iters $(ITERS) --batch-size $(BATCH) --num-layers $(LAYERS) \
	  --adapter-path $(ADAPTERS)
	@echo ">> Done. Next: make fuse"

fuse:  ## STEP 8a: merge the training result back into the model
	@echo ">> Merging the LoRA adapter into full model weights..."
	mlx_lm.fuse --model $(MODEL) --adapter-path $(ADAPTERS) --save-path $(FUSED)
	@echo ">> Done. Next: make gguf"

gguf:  ## STEP 8b: convert the model to a runnable file (needs llama.cpp source)
	@echo ">> Converting to GGUF (the format llama.cpp runs)..."
	$(PY) $(LLAMA_CPP)/convert_hf_to_gguf.py $(FUSED) --outfile $(GGUF)
	@echo ">> Compressing to $(GGUF_QUANT)..."
	llama-quantize $(GGUF) $(QGGUF) $(GGUF_QUANT)
	@echo ">> Done. Next: make serve"

# --- STEP 9: measure the model AFTER training (the 'after' number) --------- #
serve:  ## STEP 9a: serve YOUR fine-tuned model, keep running
	@echo ">> Serving your fine-tuned model on port $(PORT)."
	@echo ">> Leave this running and open a SECOND terminal for 'make eval'."
	llama-server -m $(QGGUF) --port $(PORT) --alias $(ALIAS)

eval:  ## STEP 9b: score your fine-tuned model (compare to the baseline)
	@echo ">> Scoring your fine-tuned model (this is your 'after' score)..."
	$(PY) eval/evaluate.py --data $(DATA)/test.jsonl --live --port $(PORT)

demo:  ## STEP 10: clean one messy record with your model
	$(PY) runtime/clean.py --live --port $(PORT)

review:  ## list records waiting for manual review (audit/review-queue.jsonl)
	$(PY) runtime/review.py list

all: data sanity train fuse gguf  ## do steps 5, 7 and 8 in one go (no serving)
	@echo ">> Built $(QGGUF). Now run 'make serve', then 'make eval' in a 2nd terminal."

clean:  ## delete training artifacts (keeps your data)
	rm -rf $(ADAPTERS) $(FUSED) *.gguf __pycache__ */__pycache__

distclean: clean  ## delete training artifacts AND generated data
	rm -f $(DATA)/*.jsonl
