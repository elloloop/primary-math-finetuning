#!/usr/bin/env bash
# RunPod container entrypoint script.
#
# Checks GPU availability, prints system information, creates workspace
# directories, and optionally starts TensorBoard before executing the
# container CMD.
set -euo pipefail

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
log() {
    echo "[runpod_startup] $(date '+%Y-%m-%d %H:%M:%S') $*"
}

error_exit() {
    echo "[runpod_startup] ERROR: $*" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# GPU check
# ---------------------------------------------------------------------------
log "Checking GPU availability..."
if command -v nvidia-smi &>/dev/null; then
    if nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null; then
        log "GPU detected."
    else
        log "WARNING: nvidia-smi found but no GPUs detected. Training will use CPU."
    fi
else
    log "WARNING: nvidia-smi not found. No GPU driver detected."
fi

# ---------------------------------------------------------------------------
# System information
# ---------------------------------------------------------------------------
log "=== System Info ==="

python3 --version 2>/dev/null || log "WARNING: python3 not found"

python3 - <<'PYEOF' 2>/dev/null || log "WARNING: Could not query PyTorch info"
import sys
print(f"Python:         {sys.version}")
try:
    import torch
    print(f"PyTorch:        {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version:   {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_mem / (1024**3)
            print(f"GPU {i}:          {props.name} ({vram_gb:.1f} GB)")
    else:
        print("No CUDA GPUs available.")
except ImportError:
    print("PyTorch is not installed.")
PYEOF

# ---------------------------------------------------------------------------
# Workspace directories
# ---------------------------------------------------------------------------
log "Creating workspace directories..."
mkdir -p /workspace/data \
         /workspace/outputs/models \
         /workspace/outputs/logs \
         /workspace/outputs/results \
         /workspace/outputs/visualizations \
         /workspace/cache

# Set cache environment variables (respect existing values)
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/cache}"
export HF_HOME="${HF_HOME:-/workspace/cache}"
export TORCH_HOME="${TORCH_HOME:-/workspace/cache/torch}"

log "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
log "HF_HOME=$HF_HOME"

# ---------------------------------------------------------------------------
# TensorBoard (optional)
# ---------------------------------------------------------------------------
if [[ "${START_TENSORBOARD:-false}" == "true" ]]; then
    log "Starting TensorBoard on port 6006..."
    if command -v tensorboard &>/dev/null; then
        tensorboard \
            --logdir /workspace/outputs/logs \
            --host 0.0.0.0 \
            --port 6006 \
            --reload_interval 30 &
        TENSORBOARD_PID=$!
        log "TensorBoard started (PID=$TENSORBOARD_PID)."
    else
        log "WARNING: tensorboard command not found. Skipping."
    fi
fi

# ---------------------------------------------------------------------------
# Execute CMD or fall back to bash
# ---------------------------------------------------------------------------
if [[ $# -gt 0 ]]; then
    log "Executing CMD: $*"
    exec "$@"
else
    log "No CMD provided. Dropping into bash."
    exec /bin/bash
fi
