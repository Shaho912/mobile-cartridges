# Sourced by the size-sweep SLURM scripts. Edit here once for Betty (PARCC).
export PATH=/vast/parcc/sw/bin:$PATH

# CUDA 12.8.1. Set CUDA_MODULE="" to skip if the env already provides CUDA.
CUDA_MODULE=${CUDA_MODULE-cuda/12.8.1}
if [[ -n "$CUDA_MODULE" ]]; then
    module load "$CUDA_MODULE"
fi

# CARTRIDGES_ENV may be a conda env name, a conda env path, or a venv path.
CARTRIDGES_ENV=${CARTRIDGES_ENV:-base}
eval "$(conda shell.bash hook)"
if [[ -f "$CARTRIDGES_ENV/bin/activate" && ! -d "$CARTRIDGES_ENV/conda-meta" ]]; then
    source "$CARTRIDGES_ENV/bin/activate"
elif ! conda activate "$CARTRIDGES_ENV"; then
    echo "Could not activate '$CARTRIDGES_ENV'. Find it with: conda env list" >&2
    echo "then: export CARTRIDGES_ENV=<name or path>  (and re-source this file)" >&2
    return 1 2>/dev/null || exit 1
fi

export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_$USER
export VLLM_NO_USAGE_STATS=1
export PYTHONUNBUFFERED=1
# Gemma 3 is gated on Hugging Face: request access, then `huggingface-cli login`
# once on the cluster (or export HF_TOKEN) before running the sweep.
