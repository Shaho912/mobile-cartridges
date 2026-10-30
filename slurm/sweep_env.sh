# Sourced by the size-sweep SLURM scripts. Edit here once for Betty (PARCC).
export PATH=/vast/parcc/sw/bin:$PATH

# CUDA 12.8.1. Set CUDA_MODULE="" to skip if the env already provides CUDA.
CUDA_MODULE=${CUDA_MODULE-cuda/12.8.1}
if [[ -n "$CUDA_MODULE" ]]; then
    module load "$CUDA_MODULE"
fi

eval "$(conda shell.bash hook)"
conda activate "${CARTRIDGES_ENV:-$HOME/cartridges-env}"

export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_$USER
export VLLM_NO_USAGE_STATS=1
export PYTHONUNBUFFERED=1
# Gemma 3 is gated on Hugging Face: request access, then `huggingface-cli login`
# once on the cluster (or export HF_TOKEN) before running the sweep.
