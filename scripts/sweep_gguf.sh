#!/bin/bash
# Cross-check deployable weight size with llama.cpp: HF -> GGUF bf16 -> Q8_0 and Q4_K_M.
# These files come from the original bf16 checkpoint (not the fake-quantized weights); they
# show what a real on-device format costs, embedding included, next to the analytic numbers.
#
# Needs a built llama.cpp checkout:  export LLAMA_CPP_DIR=/path/to/llama.cpp
# Usage: bash scripts/sweep_gguf.sh Qwen/Qwen3-0.6B [outputs/size_sweep/gguf]
set -eo pipefail
MODEL=${1:?usage: sweep_gguf.sh MODEL_ID [OUT_DIR]}
OUT=${2:-outputs/size_sweep/gguf}
LLAMA=${LLAMA_CPP_DIR:-}

if [[ -z "$LLAMA" || ! -f "$LLAMA/convert_hf_to_gguf.py" ]]; then
    echo "llama.cpp not found (set LLAMA_CPP_DIR to a checkout with convert_hf_to_gguf.py); skipping."
    exit 0
fi
QUANTIZE=$(command -v llama-quantize || true)
if [[ -z "$QUANTIZE" && -x "$LLAMA/build/bin/llama-quantize" ]]; then
    QUANTIZE="$LLAMA/build/bin/llama-quantize"
fi
if [[ -z "$QUANTIZE" ]]; then
    echo "llama-quantize not found (build llama.cpp first); skipping."
    exit 0
fi

SLUG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')
mkdir -p "$OUT"
SRC=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))")
python "$LLAMA/convert_hf_to_gguf.py" "$SRC" --outtype bf16 --outfile "$OUT/$SLUG-bf16.gguf"
for Q in Q8_0 Q4_K_M; do
    "$QUANTIZE" "$OUT/$SLUG-bf16.gguf" "$OUT/$SLUG-$Q.gguf" "$Q"
done

python - "$MODEL" "$OUT" "$SLUG" <<'EOF'
import json, sys
from pathlib import Path
model, out, slug = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
sizes = {q: (out / f"{slug}-{q}.gguf").stat().st_size for q in ("bf16", "Q8_0", "Q4_K_M")}
(out / f"{slug}_sizes.json").write_text(json.dumps({"model": model, "bytes": sizes}, indent=2))
for q, b in sizes.items():
    print(f"{model} GGUF {q}: {b / 1e6:.1f} MB")
EOF
