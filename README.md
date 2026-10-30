# mobile_cartridges

Research toward an **on-device personal memory system for mobile LLMs**. Built on top of [shreyansh26/cartridges](https://github.com/shreyansh26/cartridges), a single-GPU reproduction of the HazyResearch Cartridges paper (Eyuboglu et al., 2025).

## What This Is

Instead of re-prefilling a long context on every query, a compact set of learned K/V tensors (a "cartridge") is trained once to encode a document or conversation, then loaded at inference time. The long-term target is running this on-device: a local LLM compresses the day's conversation into a cartridge overnight, and loads it the next morning for persistent memory with no cloud round-trip.

**Current model:** Qwen/Qwen3-8B on a DGX B200 MIG45 GPU.

## Key Extensions

- **Multi-cartridge composition (CAS)** — distractor training so multiple cartridges coexist when concatenated at inference, achieving positive transfer across diverse patient record corpora
- **Cold/warm swap** — plug-and-play addition of new cartridges to an existing CAS set
- **Conversation cartridges** — `compress_conversation.py` compresses conversation history without vLLM
- **LongHealth integration** — MCQ benchmark eval across 8 patient records
- **Per-phase timing and GPU memory profiling**

## Setup

```bash
export CARTRIDGES_HF_MODEL_ID=Qwen/Qwen3-8B
```

See the project docs for full workflow, hyperparameters, and experiment commands.

## Model-size sweep under a 500 MB budget

Single cartridge per LongHealth patient. Goal: the smallest model whose weights plus
cartridge KV fit in 500 MB per chip and still answer well. Everything is set in
`configs/size_sweep.yaml` (models, weight quantization, slot counts, patients, teacher).

- **Weights** are fake-quantized (per-group int8/int4, quantize then dequantize to bf16) on
  every Linear and Embedding weight, including the tied embedding/lm_head. Sizes are analytic:
  params * bits / 8 plus a bf16 scale and zero point per group.
- **Teacher** is the same quantized model with the full patient record in context, and the
  cartridge is trained and evaluated on that same quantized base.
- **Bootstrap questions** are generated once per patient by vLLM and shared by every model.
  vLLM only writes question text; each run computes its own teacher logprobs.
- **No truncation.** Each run tokenizes the record and questions with its own tokenizer and
  exits with `does_not_fit` if anything exceeds the context window.
- Totals exclude runtime activations, so they are lower bounds.

On Betty, from the repo root (`/vast/parcc/sw/bin` on PATH, miniconda `base` env by default,
override with `CARTRIDGES_ENV`; edit `slurm/sweep_env.sh` if the CUDA module name differs):

```bash
# Gemma 3 is gated: request access on Hugging Face, then log in once.
huggingface-cli login

# 0. Budget table: weight MB per quantization, cartridge MB per slot count, max slots
python scripts/sweep_budget.py --config configs/size_sweep.yaml --doc-tokens

# 1. Eval sets + shared bootstrap questions (starts and stops its own vLLM server)
mkdir -p outputs/size_sweep/slurm
sbatch slurm/size_sweep_bootstrap.sbatch

# 2. Job manifest (resolves "max slots" per patient), then the array it prints
python scripts/sweep_make_jobs.py --config configs/size_sweep.yaml
sbatch --array=0-<N-1>%8 slurm/size_sweep_array.sbatch

# 3. Table + plot -> outputs/size_sweep/report/{results.md,results.csv,accuracy_vs_mb.png}
python scripts/sweep_aggregate.py --config configs/size_sweep.yaml

# Optional cross-checks
python scripts/sweep_budget.py --config configs/size_sweep.yaml --bnb   # bitsandbytes NF4 footprint (GPU)
LLAMA_CPP_DIR=~/llama.cpp bash scripts/sweep_gguf.sh Qwen/Qwen3-0.6B    # GGUF Q8_0 / Q4_K_M file sizes
```

A single run, for debugging on an interactive node
(`srun --partition=b200-mig45 --gres=gpu:1 --pty bash`):

```bash
python scripts/sweep_run.py --config configs/size_sweep.yaml \
    --model Qwen/Qwen3-0.6B --quant int4 --mode cartridge --slots 512 --kv-eval bf16 int8 --patient 1
```

Runs are resumable: a run with an existing `result.json` is skipped unless `--force` is passed.
Outputs go to `outputs/size_sweep/runs/<model>/<quant>/<run>/lh_pXX/`. Teacher supervision is
cached per model, quantization, and patient under `outputs/size_sweep/supervision/`.
