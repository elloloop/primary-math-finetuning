#!/usr/bin/env bash
set -euo pipefail

echo "=== System Info ==="
python --version
python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM(GB): {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f}")
PY

mkdir -p /workspace/{data,outputs/{models,logs,results,visualizations},cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/workspace/cache}
export HF_HOME=${HF_HOME:-/workspace/cache}

if [[ "${START_TENSORBOARD:-false}" == "true" ]]; then
  tensorboard --logdir /workspace/outputs/logs --host 0.0.0.0 --port 6006 &
fi

exec "$@"
