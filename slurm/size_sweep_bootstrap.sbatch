#!/bin/bash
# Step 1 of the size sweep: eval sets + shared bootstrap questions (one vLLM server, one job).
#SBATCH --job-name=sweep-bootstrap
#SBATCH --partition=b200-mig45
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/size_sweep/slurm/%x_%j.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/sweep_env.sh
set -u

CONFIG=${SWEEP_CONFIG:-configs/size_sweep.yaml}
export CARTRIDGES_MODEL_ID=${CARTRIDGES_MODEL_ID:-Qwen/Qwen3-4B}
# Free port by default: other jobs on a shared node may already hold 8000.
PORT=${CARTRIDGES_VLLM_PORT:-$(python -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1])")}
API_KEY=cartridges-local

# Own process group, so cleanup also stops vLLM's EngineCore child, which otherwise keeps
# the GPU and any stdout pipe (e.g. `| tee`) open after this script exits.
setsid python scripts/serve_vllm.py --model "$CARTRIDGES_MODEL_ID" --port "$PORT" \
    --api-key "$API_KEY" --gpu-memory-utilization 0.85 &
VLLM_PID=$!
trap 'kill -- -$VLLM_PID 2>/dev/null || true' EXIT

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

python scripts/sweep_bootstrap.py --config "$CONFIG" --base-url "http://127.0.0.1:$PORT/v1"
