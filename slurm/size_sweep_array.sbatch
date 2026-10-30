#!/bin/bash
# Step 2 of the size sweep: one array task per line of outputs/size_sweep/jobs/manifest.txt.
# No vLLM needed: every run computes its own teacher logprobs with its own quantized model.
# Submit with the command printed by scripts/sweep_make_jobs.py, e.g.
#   sbatch --array=0-214%8 slurm/size_sweep_array.sbatch
#SBATCH --job-name=size-sweep
#SBATCH --partition=b200-mig45
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/size_sweep/slurm/%x_%A_%a.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/sweep_env.sh
set -u

CONFIG=${SWEEP_CONFIG:-configs/size_sweep.yaml}
MANIFEST=${SWEEP_MANIFEST:-outputs/size_sweep/jobs/manifest.txt}

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
if [[ -z "$LINE" ]]; then
    echo "No manifest line for task $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi
read -r -a RUN_ARGS <<< "$LINE"
echo "task $SLURM_ARRAY_TASK_ID: ${RUN_ARGS[*]}"
python scripts/sweep_run.py --config "$CONFIG" "${RUN_ARGS[@]}" --device cuda:0
