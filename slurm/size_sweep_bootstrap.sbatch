#!/bin/bash
# Step 1 of the size sweep: eval sets + shared bootstrap questions (one vLLM server, one job).
#SBATCH --job-name=sweep-bootstrap
#SBATCH --partition=b200-mig45
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/size_sweep/slurm/%x_%j.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/sweep_env.sh
set -u

CONFIG=${SWEEP_CONFIG:-configs/size_sweep.yaml}
export CARTRIDGES_MODEL_ID=${CARTRIDGES_MODEL_ID:-Qwen/Qwen3-4B}
PORT=${CARTRIDGES_VLLM_PORT:-8000}
API_KEY=cartridges-local

python scripts/serve_vllm.py --model "$CARTRIDGES_MODEL_ID" --port "$PORT" \
    --api-key "$API_KEY" --gpu-memory-utilization 0.85 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
    if curl -sf -H "Authorization: Bearer $API_KEY" "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM server exited during startup" >&2
        exit 1
    fi
    sleep 10
done
curl -sf -H "Authorization: Bearer $API_KEY" "http://127.0.0.1:$PORT/v1/models" >/dev/null \
    || { echo "vLLM server not ready after 20 minutes" >&2; exit 1; }

python scripts/sweep_bootstrap.py --config "$CONFIG"
