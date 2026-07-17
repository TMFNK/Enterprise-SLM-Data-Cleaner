# Enterprise SLM Data Cleaner

Enterprise-grade version of [Local-SLM-Data-Cleaner](https://github.com/TMFNK/Local-SLM-Data-Cleaner):
a small language model (SLM), fine-tuned entirely on synthetic data, that
normalizes unclean SAP-style master data to a documented house convention. It is built
to run in secure, air-gapped enterprise environments.

Where the original repo is a beginner-friendly demo for a Mac laptop, this
version targets production deployments: editable client-specific convention specs,
a (manual) review queue with an append-only audit trail, containerized offline
serving with vendored model weights, and eval-gated releases.

---

## What this project does

Every company running SAP knows the problem: the same supplier exists three
times with three spellings, countries are written as "Germany", "Deutschland"
or "DE", dates and amounts come in German and US formats, and missing values
are encoded in five different ways. Cleaning this by hand is slow, as writing rules
for every possible mistake is tiresome.

This project takes a different route. Your data standard is written down
once, as a readable document. From that standard, the system invents
thousands of practice examples (dirty record IN, clean record OUT) and
teaches a small AI model the standard. The result is a model that cleans
records the way your rules would, and also handles the misspellings and
odd formats no rule anticipated.

The decisive point: **your data never leaves your machine.** There is no
cloud service, no API subscription, no data transfer to anyone.
The model is a file on your own hardware, small enough to run on
an ordinary office machine, and it works without Wi-Fi, or the network cable.

## Für Entscheider (kurz gefasst)

Sensible Stammdaten an eine ausländische Cloud-KI zu senden, ist unter der
DSGVO oft keine Option, und für viele Häuser auch keine Frage der Rechtslage,
sondern der Grundhaltung. Dieses Projekt zeigt den anderen Weg:

- **Alles läuft lokal.** Das Modell ist eine Datei auf Ihrer eigenen Hardware.
  Kein Cloud-Dienst, kein Abo, keine Token-Kosten, keine Datenweitergabe. Auf
  Wunsch komplett vom Netz getrennt (Air-Gap), nachweisbar durch die
  Container-Konfiguration, nicht nur durch Versprechen.
- **Trainiert ausschließlich auf erfundenen Daten.** Kein einziger echter
  Datensatz wird für das Training verwendet. Ihre Daten kommen erst zur
  Laufzeit ins Spiel und bleiben im Haus.
- **Jede Änderung ist nachvollziehbar.** Ein unveränderliches Protokoll
  (Audit-Log) hält für jeden Datensatz fest, was geändert wurde, mit welcher
  Konventionsversion und mit welchem Modellstand. Unsichere Fälle gehen in
  eine manuelle Prüfschlange, nie stillschweigend durch.
- **Das Basismodell ist austauschbar.** Wer kein chinesisches Modell einsetzen
  möchte, wählt ein europäisches (z. B. Ministral-8B oder Mistral Nemo von
  Mistral AI in Frankreich, Teuken-7B von Fraunhofer, gefördert vom BMWK, oder
  EuroLLM aus einem EU-Projekt) oder ein US-Modell mit MIT-Lizenz. Details in
  der Tabelle unten.

Beratung und Umsetzung: [mbitai.com](https://www.mbitai.com).

---

## How it works, step by step (no ML knowledge required)

1. **The rules.** Your data standard (allowed country codes, legal forms,
   date formats, how missing values are written) lives in one readable file
   client-specific: [conventions/default.yaml](conventions/default.yaml). A data
   steward can open and edit it; no programming involved. Switching clients
   means switching files, not rewriting software.

2. **Practice data, invented from the rules.** The system generates
   thousands of fake records (invented company names, invented IBANs), makes
   them dirty in realistic ways, and computes the correct cleaned version
   from the rulebook. No real data is used at any point, which is why the
   training itself is GDPR-uncritical.

3. **Training, on ordinary hardware.** A small open-source model (about 1 GB;
   see the model table below) practices on those examples until it reliably
   produces clean records. This runs on a normal laptop in under an hour;
   no GPU cluster, no cloud.

4. **Operation with a double check.** In production, every record the model
   cleans is immediately re-checked against the rulebook. If the answer
   violates any rule, the deterministic rule engine takes over as a safety
   net. The model adds value on the messy edge cases; the rules guarantee a
   floor it can never fall below.

5. **Nothing uncertain slips through.** Records the system is not sure about
   (rule violations, low model confidence) go to a manual review queue. A
   person decides; the decision is recorded. `make review` lists what is
   waiting.

6. **Everything is on the record.** Every cleaning decision is appended to an
   audit log: input, output, every single change, confidence, the exact
   version (hash) of the model weights and of the convention file. The log is
   append-only: entries are never edited or deleted, resolutions are added as
   new entries. This is what your auditor and your Datenschutzbeauftragter
   will ask for.

7. **Sealed delivery.** For production, everything ships as one container
   that is run with its network stack removed (`--network none`). The
   container refuses to start if the model weights do not match the
   fingerprint pinned in version control, so nobody can swap the model
   unnoticed. Records go in and out through a mounted folder only. See
   [deploy/README.md](deploy/README.md).

8. **A quality gate on every change.** The repository carries a suite of
   pinned test cases, including adversarial ones (is "Bavaria" wrongly
   "corrected" to a country? is "mbH" recognized as GmbH? is a US date
   misread as a German one?). Continuous integration blocks any change that
   alters this documented behavior. Releases are gated by measured accuracy,
   not by hope.

9. **Semantic alias resolution (optional).** An embedding soft-normalizer can
   sit in the cleaning pipeline to catch near-misses the alias map never
   listed. Off by default; see [Semantic embedding layer](#semantic-embedding-layer)
   below.

---

## Cleaning pipeline architecture

Every controlled-vocabulary field (country, legal form, status, currency,
unit) is normalized in a fixed order. Later stages only run when earlier ones
miss. This way, documented aliases stay authoritative and the optional embedding
layer never overrides a known mapping:

```text
raw value
  → blank / sentinel?     → null
  → exact controlled set? → keep
  → deterministic alias?  → canonical   (conventions/*.yaml)
  → embedding near-miss?  → canonical   (optional, USE_EMBEDDINGS=1)
  → otherwise             → as-is       (grounding: flagged, not invented)
```

**Why this order.** Putting embeddings _after_ the alias map preserves the
house standard: "germany" → `DE` always comes from the YAML, never from a
similarity guess. Embeddings only help with novel paraphrases
("Nederlands", "Federal Republic of Germany") that nobody thought to list.

**Grounding guarantee.** Values below the per-field similarity threshold
pass through unchanged. Regions and true unknowns must not be "helpfully"
corrected: `Bavaria` stays `Bavaria`, `Atlantis` stays `Atlantis`, the typo
`GmbbH` stays `GmbbH`. The adversarial suite pins these cases.

**Labels stay honest.** Synthetic near-miss corruption runs only when
`USE_EMBEDDINGS=1`, so `normalize_record()` can reverse it. Without the flag,
training data never teaches the model to pass through a value that production
would clean.

---

## Semantic embedding layer

Activate with `USE_EMBEDDINGS=1` (off by default; no overhead when unset).

```bash
pip install sentence-transformers>=3.0
make embed                          # download + cache the embedding model
USE_EMBEDDINGS=1 make data          # near-miss examples get correct labels
USE_EMBEDDINGS=1 make eval-gate     # include semantic_alias adversarial cases
USE_EMBEDDINGS=1 make demo          # use embeddings at inference
```

Thresholds are per field under `embedding_thresholds:` in
[conventions/default.yaml](conventions/default.yaml). The resolver compares
the unknown string against **canonical codes and every alias key** (readable
anchors). Matching bare ISO codes like `DE` alone is not enough: short codes
carry almost no semantics for an embedding model.

### Default embedding model

| Model                                             | Size  | Role                                      | Notes                                                                                                                                  |
| ------------------------------------------------- | ----- | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | ~570M | **Default** in `core/embedding_lookup.py` | Multilingual (100+ languages), strong on short phrases, loads via `sentence-transformers`. ~2.2 GB download, ~1.2 GB RAM at inference. |

`bge-m3` is the right default for this stack: it runs on CPU or Apple Silicon
MPS, needs no NVIDIA GPU, and handles the German/English/French mix typical
of European master data.

### Alternative embedding models

The cleaner only needs a text → dense vector encoder with cosine similarity.
Any of the following can replace `BAAI/bge-m3` in
`core/embedding_lookup.py` (one-line model id change); re-tune
`embedding_thresholds:` and re-run `make embed` plus the adversarial suite
before shipping.

| Model                                                                                                                                             | Size          | Best when                                                                       | Caveats                                                                                                                                                         |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [nvidia/Nemotron-3-Embed-1B-NVFP4](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-NVFP4)                                                       | 1.14B (NVFP4) | NVIDIA GPU fleets; multilingual retrieval quality; commercial use (OpenMDW-1.1) | Quantized for **vLLM** on NVIDIA GPUs; not a drop-in for `sentence-transformers`. Prefer the BF16 sibling below on CPU/Mac, or wire vLLM as the encode backend. |
| [nvidia/Nemotron-3-Embed-1B-BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16)                                                         | 1.14B         | Same family, easier local swap via Transformers / Sentence Transformers         | Heavier than bge-m3; validate thresholds (esp. grounding: Bavaria must not become DE).                                                                          |
| [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base)                                                             | ~278M         | Smaller multilingual footprint                                                  | May need the `query:` / `passage:` prefixes E5 expects; retune thresholds.                                                                                      |
| [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | ~120M         | Fast CPU / laptop demos                                                         | Weaker on rare country synonyms; raise thresholds carefully.                                                                                                    |

**Practical recommendation.** Keep **bge-m3** for Mac / air-gapped CPU boxes.
Evaluate **Nemotron-3-Embed-1B** when you already run NVIDIA inference and want
stronger multilingual retrieval. If you want to use the
[NVFP4](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-NVFP4) checkpoint
with vLLM in the data center, or the
[BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16) weights where
`sentence-transformers` is enough. Always re-check the grounding cases after
a swap.

---

## Choosing the base model (DIGITAL SOVEREIGNTY)

The demo default is Qwen3-0.6B, an Apache-2.0 open-source model originally
published by Alibaba. Two frank statements about that:

**Objectively, the origin of the weights is not a data-protection issue in
this architecture.** The weights are a static file of numbers, fine-tuned by
us on synthetic data, running in a container that has no network.
There is no channel for data to flow anywhere, regardless of who trained
the base model.

**Commercially and politically, the choice still matters.** Procurement,
works councils and security officers have preferences, and "the AI is
Chinese" can end a conversation in a German enterprise regardless of the
technical facts. This stack is therefore deliberately **model-agnostic**: the
convention specs, synthetic data, eval suite, audit trail and container do
not care which base model sits inside! Swapping means retraining (cheap, as the
training data is already generated) and re-running the same eval gate.

| Base model                                                                                | Size | Origin                          | License     | Notes                                                                                                                                                       |
| ----------------------------------------------------------------------------------------- | ---- | ------------------------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)                                      | 0.6B | Alibaba (CN)                    | Apache-2.0  | Demo default: smallest, fastest, runs on 8 GB laptops                                                                                                       |
| [Teuken-7B-instruct](https://huggingface.co/openGPT-X/Teuken-7B-instruct-commercial-v0.4) | 7B   | **Fraunhofer / OpenGPT-X (DE)** | Apache-2.0  | German-made, BMWK-funded, all 24 EU languages. Nice sovereignty pick, but needs a bigger training machine (≥32 GB or a GPU)                                 |
| [EuroLLM-1.7B](https://huggingface.co/utter-project/EuroLLM-1.7B-Instruct)                | 1.7B | **EU project**                  | Apache-2.0  | European, all official EU languages, still small footprint                                                                                                  |
| [Ministral-8B-Instruct-2410](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410) | 8B   | **Mistral AI (FR)**             | Apache-2.0  | French-made edge model; LoRA-trains on a 16 GB Mac (~4.5 GB at 4-bit). Multilingual incl. DE. Strong EU pick when 0.6B is too small but Teuken is too heavy |
| [Mistral Nemo Instruct](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)      | 12B  | **Mistral AI (FR)**             | Apache-2.0  | Mistral+NVIDIA collab; better instruct quality than the 7–8B class. Plan for ≥16 GB RAM at 4-bit (~7 GB weights)                                            |
| [Phi-4-mini](https://huggingface.co/microsoft/Phi-4-mini-instruct)                        | 3.8B | Microsoft (US)                  | MIT         | Most permissive license of the field; strong quality, mid-sized                                                                                             |
| [Gemma 3 1B](https://huggingface.co/google/gemma-3-1b-it)                                 | 1B   | Google (US)                     | Gemma Terms | Capable, but _not_ pure open source: Google's terms attach a use policy                                                                                     |
| [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)                             | 3B   | Hugging Face (US/FR)            | Apache-2.0  | Fully documented training recipe, strong transparency story                                                                                                 |

Practical recommendation for EU customers:

- **EuroLLM-1.7B**: smallest European footprint; still fits the 8 GB demo laptop.
- **Ministral-8B-Instruct-2410**: the sweet spot for _made in the EU_ without a
  server room: French company, Apache-2.0, trains on an ordinary 16 GB Mac, and
  enough capacity for messy German master-data edge cases that a 0.6B model
  struggles with.
- **Mistral Nemo 12B**: same sovereignty story with more headroom; pick it when
  you have ≥16 GB RAM and want better zero-shot quality before fine-tuning.
- **Teuken-7B**: where _made in Germany_ carries the most weight and a bigger
  training machine (≥32 GB or a GPU) is available.

Phi-4-mini remains the US quality/license fallback if procurement rules out
European weights entirely.

To swap, rerun the pipeline with a different `MODEL_PRESET`. Run `make clean`
first so adapters and GGUF files from the previous model are not reused:

```bash
make list-models        # show available model presets
make MODEL_PRESET=minicpm5-1b clean model data train fuse gguf serve eval

# All pipeline commands after switch respect the preset:
make MODEL_PRESET=minicpm5-1b train    # re-run training only
make MODEL_PRESET=minicpm5-1b eval     # re-run evaluation
```

Or set it once for the session:

```bash
export MODEL_PRESET=minicpm5-1b
make clean model data train fuse gguf serve eval
```

Available presets:

| Preset         | Base model                                                | Size | GGUF repo (baseline)                                               | Quant    | ALIAS                  |
| -------------- | --------------------------------------------------------- | ---- | ------------------------------------------------------------------ | -------- | ---------------------- |
| `qwen3-0.6b`   | [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)      | 0.6B | `Qwen/Qwen3-0.6B-GGUF`                                             | `Q8_0`   | `qwen3-0.6b-cleaner`   |
| `qwen3.5-0.8b` | [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)  | 0.8B | `unsloth/Qwen3.5-0.8B-GGUF` (community; no official Qwen GGUF yet) | `Q8_0`   | `qwen3.5-0.8b-cleaner` |
| `minicpm5-1b`  | [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) | 1B   | `openbmb/MiniCPM5-1B-GGUF`                                         | `Q4_K_M` | `minicpm5-1b-cleaner`  |

Each preset sets four variables together. You can still override any individually:

```bash
make model train fuse gguf MODEL_PRESET=minicpm5-1b GGUF_QUANT=Q8_0
```

(One caveat: each architecture must be supported by the MLX
trainer and llama.cpp! Qwen, Gemma, Phi, Llama-family, SmolLM, MiniCPM, and
Mistral (including Ministral and Nemo) are all supported today.)

---

## Layout

```bash
core/          the convention engine (loads the YAML spec, single source of truth)
               core/embedding_lookup.py (optional) post-alias soft-normalizer
               (default encoder: BAAI/bge-m3; swap-friendly)
conventions/   editable client-specific convention specs (YAML)
               embedding_thresholds: + alias maps feed the resolver
synth/         synthetic messy->clean data generator (no real data, ever)
eval/          eval harness + pinned adversarial suite
               (legal forms, formats, grounding, semantic_alias)
runtime/       clean service: model -> validate -> algorithm safety net + audit + review
deploy/        air-gapped container: Containerfile, entrypoint, security notes
train/         MLX LoRA fine-tuning notes
.github/       CI eval gate: regressions cannot merge
Makefile       every pipeline step as `make <command>` (see `make help`)
```

## Quick start (pipeline)

```bash
make setup           # install Python deps + mlx-lm
make data            # generate synthetic train/valid/test data
make sanity          # verify the data against the rule-based algorithm (~100%)
make eval-gate       # regression gate: sanity + adversarial suites must be 100%
make train           # LoRA fine-tune the base model (Apple Silicon / MLX)
make fuse gguf       # package the trained model for serving
make pin-model       # vendor the weights + pin their hash for the container
make serve           # serve it locally...
make eval            # ...and measure the before/after lift
make review          # list records waiting for manual review

# optional: embedding soft-normalizer (raises score on near-miss aliases)
pip install sentence-transformers>=3.0
make embed
USE_EMBEDDINGS=1 make data eval-gate
```

Client-specific convention: `make data CONVENTION=conventions/<client>.yaml`

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
