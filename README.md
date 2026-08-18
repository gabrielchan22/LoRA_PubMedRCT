# LoRA From Scratch: Biomedical Sentence Classification

A from-scratch PyTorch implementation of LoRA (Low-Rank Adaptation), used
to fine-tune a pretrained transformer on the PubMed-RCT sentence
classification task, benchmarked directly against full fine-tuning of the
same model.

## What this shows

Most LoRA usage in the wild is `peft.LoraConfig` — a config object over a
library call. This project implements the adapter mechanics directly
(`lora_layers.py`) and uses that implementation, rather than a library, to
run a controlled comparison against full fine-tuning: same base model,
same data, same shared training loop — with accuracy, GPU memory, training
time, and checkpoint size all measured for every run.

## Task & data

[PubMed-RCT](https://huggingface.co/datasets/armanc/pubmed-rct20k):
classify each sentence of a biomedical abstract as `BACKGROUND`,
`OBJECTIVE`, `METHODS`, `RESULTS`, or `CONCLUSIONS` — a real structuring
task for unstructured medical literature (177k train / 29.7k validation /
29.6k test sentences).

**Model:** `roberta-base` (125M params), fine-tuned two ways:
1. **Full fine-tuning** — every parameter trainable (baseline)
2. **LoRA** — base frozen, low-rank adapters (`lora_A`, `lora_B`) injected
   into the `query`/`value` attention projections of every layer

## Project structure

| File | Purpose |
|---|---|
| `lora_layers.py` | `LoRALinear` (the adapter itself) + `inject_lora` (recursively wires it into a real model) |
| `data.py` | Loads and tokenizes PubMed-RCT into PyTorch `DataLoader`s |
| `model.py` | Builds the full fine-tune model and the LoRA-injected model |
| `metrics.py` | Shared measurement harness: trainable param count, GPU memory, wall-clock time, checkpoint size, accuracy/macro-F1 |
| `train.py` | One training loop used by both experiment types |
| `run_experiments.py` | Orchestrates the baseline run + the rank/scaling ablation sweep, writes `results.csv` |
| `plots.py` | Accuracy-vs-rank and memory-vs-rank charts from `results.csv` |

## The ablation

Sweeps LoRA rank `r ∈ {4, 8, 16, 32, 64}` against two adapter-scaling
conventions:
- `alpha/r` — the original LoRA paper's convention
- `alpha/√r` — proposed by follow-up work ("rank-stabilized LoRA") as a
  more precise way to hold the adapter's output magnitude constant across
  ranks (empirically verified in this project's development — see writeup)

## Key implementation details worth knowing

- **Asymmetric initialization is load-bearing.** `lora_A` is Kaiming-initialized,
  `lora_B` is zero-initialized. Zero-initializing both gives zero gradient
  for *both* matrices — the adapter never leaves zero. (Verified directly:
  see `lora_layers.py`'s test suite in the dev history.)
- **`merge()` is exact, not approximate** — folding the adapter into the
  base weight is a pure algebraic rearrangement, verified to match
  pre-merge output with zero numerical difference beyond float roundoff.
- **Only `query`/`value` are adapted**, matching the original paper's
  finding that this captures most of the benefit at a fraction of the
  parameter cost of adapting every linear layer.

## Running it

```bash
pip install -r requirements.txt
python run_experiments.py   # needs a GPU + internet access for model/dataset download
python plots.py
```

Built and tuned for a single free-tier Colab T4 GPU.

## Results

_Fill in after running `run_experiments.py`:_

| Method | Rank | Alpha mode | Accuracy | Macro F1 | Wall-clock (s) | Peak memory (MB) | Checkpoint size (MB) | Trainable % |
|---|---|---|---|---|---|---|---|---|
| Full fine-tune | – | – | | | | | | 100% |
| LoRA | 8 | alpha/r | | | | | | |
| LoRA | 8 | alpha/√r | | | | | | |
| ... | | | | | | | | |

![Accuracy vs rank](accuracy_vs_rank.png)
![Memory vs rank](memory_vs_rank.png)
