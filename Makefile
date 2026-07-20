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
#   MODEL_PRESET  model family shortcut: qwen3-0.6b, qwen3.5-0.8b, minicpm5-1b
#   MODEL         base model to fine-tune (auto-downloads)
#   GGUF_HF       stock GGUF repo for the zero-shot baseline
#   GGUF_QUANT    quant level to download/build (Q8_0, Q4_K_M, ...)
#   N             number of synthetic examples
#   SEED          random seed (same seed = same data)
#   ITERS         training steps
#   BATCH         training batch size
#   LAYERS        how many layers LoRA touches
#   PORT          local server port
#   ALIAS         model name the eval/clean scripts look for
#   LLAMA_CPP     path to a llama.cpp source checkout (only for `make gguf`)
#   CONVENTION    convention spec file (per-client: conventions/<client>.yaml)
MODEL_PRESET ?= qwen3-0.6b

# Model presets — set MODEL_PRESET to quickly switch between supported models.
# Each preset sets MODEL, GGUF_HF, GGUF_QUANT, and ALIAS. Override any
# individually on the command line (e.g. `make train MODEL=...`).
# Switching presets? Run `make clean` first so adapters/GGUF from the old model
# are not reused.
ifeq ($(MODEL_PRESET),qwen3-0.6b)
  MODEL      ?= Qwen/Qwen3-0.6B
  GGUF_HF    ?= Qwen/Qwen3-0.6B-GGUF
  GGUF_QUANT ?= Q8_0
  ALIAS      ?= qwen3-0.6b-cleaner
else ifeq ($(MODEL_PRESET),qwen3.5-0.8b)
  MODEL      ?= Qwen/Qwen3.5-0.8B
  GGUF_HF    ?= unsloth/Qwen3.5-0.8B-GGUF
  GGUF_QUANT ?= Q8_0
  ALIAS      ?= qwen3.5-0.8b-cleaner
else ifeq ($(MODEL_PRESET),minicpm5-1b)
  MODEL      ?= openbmb/MiniCPM5-1B
  GGUF_HF    ?= openbmb/MiniCPM5-1B-GGUF
  GGUF_QUANT ?= Q4_K_M
  ALIAS      ?= minicpm5-1b-cleaner
else
  $(error Unknown MODEL_PRESET '$(MODEL_PRESET)'. Run 'make list-models'.)
endif

N          ?= 1000
SEED       ?= 0
ITERS      ?= 1000
BATCH      ?= 4
LAYERS     ?= 8
PORT       ?= 8080
DATA       ?= data
ADAPTERS   ?= adapters
FUSED      ?= fused
GGUF       ?= $(ALIAS).gguf
# GGUF quant file uses lowercase version of GGUF_QUANT in its name.
# E.g. Q8_0 -> -q8_0, Q4_K_M -> -q4_k_m.
QGGUF_SUFFIX = $(shell echo $(GGUF_QUANT) | tr 'A-Z' 'a-z')
QGGUF      ?= $(ALIAS)-$(QGGUF_SUFFIX).gguf
LLAMA_CPP  ?= ../llama.cpp
PY         ?= python3
CONVENTION ?= conventions/default.yaml
export CONVENTION
MODELS_DIR ?= models

.DEFAULT_GOAL := help
.PHONY: help list-models setup model data sanity adversarial embed eval-gate eval-adversarial \
        baseline-serve baseline train fuse gguf \
        serve eval demo review pin-model verify-model all clean distclean \
        privacy-check check-balance fixtures report-oracle eval-unseen eval-gold

help:  ## show this list of commands
	@echo "Enterprise SLM Data Cleaner: commands (run them in this order):"
	@echo ""
	@echo "Model preset: $(MODEL_PRESET)  (run 'make list-models' for options)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

list-models:  ## show available model presets
	@echo "Available MODEL_PRESET values:"
	@echo ""
	@echo "  qwen3-0.6b     Qwen3-0.6B  (default) — 600M params, ~1 GB, Apache-2.0"
	@echo "  qwen3.5-0.8b   Qwen3.5-0.8B          — 800M params, ~1.5 GB, Apache-2.0"
	@echo "  minicpm5-1b    MiniCPM5-1B           — 1B params, SOTA in class, Apache-2.0"
	@echo ""
	@echo "Usage:  make MODEL_PRESET=minicpm5-1b <command>"
	@echo "Set it once:  export MODEL_PRESET=minicpm5-1b"
	@echo "Switch preset:  make clean  (drops adapters/GGUF from the previous model)"

privacy-check:  ## scan fixtures/real for committed extracts (see docs/PRIVACY.md)
	$(PY) scripts/privacy_check.py

fixtures:  ## rebuild pinned gold + unseen holdout (commit intentionally)
	$(PY) scripts/build_fixtures.py

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
model:  ## STEP 4: download the base model from Hugging Face
	@echo ">> Model preset: $(MODEL_PRESET)"
	@echo ">> Downloading $(MODEL) (first time only, no login needed)..."
	@echo ">> It caches in ~/.cache/huggingface so later steps are instant."
	$(PY) -c "from mlx_lm import load; load('$(MODEL)'); print('model ready')"
	@echo ">> Done. Next: make data"

# --- STEP 5: make the training data ---------------------------------------- #
data:  ## STEP 5a: generate synthetic train/valid/test data (N, SEED) under data/ only
	@echo ">> Generating $(N) synthetic messy->clean examples..."
	$(PY) synth/generate.py --n $(N) --out $(DATA) --seed $(SEED)
	@echo ">> Done. Next: make sanity (and make check-balance before train)"

sanity:  ## STEP 5b: check the data is correct (should say 100%)
	@echo ">> Checking the held-out test split against the rule-based algorithm..."
	$(PY) eval/evaluate.py --data $(DATA)/test.jsonl --algorithm
	@echo ">> If the numbers are ~100%, the data is good. Next: make baseline-serve"

check-balance:  ## fail if train set lacks recordType / vocab coverage
	$(PY) train/check_balance.py --data $(DATA)/train.jsonl

eval-gold:  ## score pinned gold fixtures with the oracle
	$(PY) eval/evaluate.py --data fixtures/gold.jsonl --algorithm --label oracle \
	  --min-score 100

eval-unseen:  ## score unseen-noise holdout with the oracle
	$(PY) eval/evaluate.py --data fixtures/holdout_unseen_noise.jsonl --algorithm --label oracle

report-oracle:  ## write reports/oracle-gold.md from pinned gold
	@mkdir -p reports
	$(PY) eval/evaluate.py --data fixtures/gold.jsonl --algorithm --label oracle \
	  --report reports/oracle-gold.md --min-score 100

adversarial:  ## build the pinned adversarial eval suite (fails loudly on regressions)
	$(PY) eval/adversarial.py --out $(DATA)/adversarial.jsonl

embed:  ## pre-compute embeddings + cache bge-m3 (optional, for USE_EMBEDDINGS=1)
	USE_EMBEDDINGS=1 PYTHONPATH=core $(PY) -c "\
import sentence_transformers; \
from convention_spec import normalize_record; \
assert normalize_record({'country': 'BRD'})[0]['country'] == 'DE'; \
assert normalize_record({'country': 'Nederlands'})[0]['country'] == 'NL'; \
assert normalize_record({'country': 'Bavaria'})[0]['country'] == 'Bavaria', \
    'region must not map to a country'; \
assert normalize_record({'country': 'Atlantis'})[0]['country'] == 'Atlantis'; \
assert normalize_record({'legalForm': 'GmbbH'})[0]['legalForm'] == 'GmbbH'; \
print('bge-m3 cached and ready')"

eval-gate: privacy-check data adversarial eval-gold  ## CI gate: privacy + gold + synth + adversarial
	$(PY) core/convention_spec.py
	$(PY) eval/evaluate.py --data $(DATA)/test.jsonl --algorithm --min-score 100
	$(PY) eval/evaluate.py --data $(DATA)/adversarial.jsonl --algorithm --min-score 100
	$(PY) eval/evaluate.py --data fixtures/holdout_unseen_noise.jsonl --algorithm --min-score 100
	@echo ">> Eval gate passed."

eval-adversarial:  ## score the served model on the adversarial suite (per category)
	$(PY) eval/evaluate.py --data $(DATA)/adversarial.jsonl --live --port $(PORT) --model-name $(ALIAS)

# --- STEP 6: measure the model BEFORE training (the 'before' number) ------- #
baseline-serve:  ## STEP 6a: serve the STOCK model (downloads it fresh), keep running
	@echo ">> Downloading + serving the untrained stock model on port $(PORT)."
	@echo ">> Leave this running and open a SECOND terminal for 'make baseline'."
	llama-server -hf $(GGUF_HF):$(GGUF_QUANT) --port $(PORT) --alias $(ALIAS)

baseline:  ## STEP 6b: score the stock model (run in the 2nd terminal)
	@echo ">> Scoring the untrained model (this is your 'before' score)..."
	$(PY) eval/evaluate.py --data fixtures/gold.jsonl --live --port $(PORT) --model-name $(ALIAS) \
	  --label "base SLM" --report reports/base-gold.md

# --- STEP 7: fine-tune ------------------------------------------------------ #
train: check-balance  ## STEP 7: fine-tune (balance gate must pass first)
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

eval:  ## STEP 9b: score your fine-tuned model on pinned gold (compare to baseline)
	@echo ">> Scoring your fine-tuned model (this is your 'after' score)..."
	$(PY) eval/evaluate.py --data fixtures/gold.jsonl --live --port $(PORT) --model-name $(ALIAS) \
	  --label "fine-tuned SLM" --report reports/ft-gold.md
	@echo ">> Also score adversarial + unseen locally if desired:"
	@echo "   make eval-adversarial   and   make eval-unseen (oracle) / live with --data fixtures/holdout_unseen_noise.jsonl"

demo:  ## STEP 10: clean one messy record with your model
	$(PY) runtime/clean.py --live --port $(PORT) --model-name $(ALIAS)

review:  ## list records waiting for manual review (audit/review-queue.jsonl)
	$(PY) runtime/review.py list

pin-model:  ## vendor the trained GGUF into models/ and pin its sha256 (after make gguf)
	@mkdir -p $(MODELS_DIR)
	@if [ -f $(QGGUF) ]; then cp -f $(QGGUF) $(MODELS_DIR)/; fi
	@ls $(MODELS_DIR)/*.gguf >/dev/null 2>&1 \
	  || { echo "!! no .gguf in $(MODELS_DIR)/ and no $(QGGUF) here. Run 'make gguf' first."; exit 1; }
	@cd $(MODELS_DIR) && shasum -a 256 *.gguf > MANIFEST.sha256 && cat MANIFEST.sha256
	@echo ">> Pinned. Commit $(MODELS_DIR)/MANIFEST.sha256 (weights stay gitignored)."

verify-model:  ## check vendored weights in models/ against the pinned manifest
	@cd $(MODELS_DIR) && shasum -a 256 -c MANIFEST.sha256

all: data sanity train fuse gguf  ## do steps 5, 7 and 8 in one go (no serving)
	@echo ">> Built $(QGGUF). Now run 'make serve', then 'make eval' in a 2nd terminal."

clean:  ## delete training artifacts (keeps your data)
	rm -rf $(ADAPTERS) $(FUSED) *.gguf __pycache__ */__pycache__

distclean: clean  ## delete training artifacts AND generated data
	rm -f $(DATA)/*.jsonl
