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
EXPERIMENT="${EXPERIMENT:-default}"
export EXPERIMENT
log "Experiment: $EXPERIMENT"

log "Creating workspace directories..."
mkdir -p /workspace/data \
         /workspace/outputs/models/"$EXPERIMENT" \
         /workspace/outputs/logs/"$EXPERIMENT" \
         /workspace/outputs/results/"$EXPERIMENT" \
         /workspace/outputs/visualizations \
         /workspace/cache

# Set cache environment variables (respect existing values)
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/cache}"
export HF_HOME="${HF_HOME:-/workspace/cache}"
export TORCH_HOME="${TORCH_HOME:-/workspace/cache/torch}"

log "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
log "HF_HOME=$HF_HOME"

# ---------------------------------------------------------------------------
# Clone training data repo (skip if already present on the volume)
# ---------------------------------------------------------------------------
DATA_REPO_URL="${DATA_REPO_URL:-git@github.com:elloloop/maths-questions-database.git}"
DATA_DIR="/workspace/data/maths-questions-database"

if [[ -d "$DATA_DIR/.git" ]]; then
    log "Training data already present at $DATA_DIR, pulling latest..."
    git -C "$DATA_DIR" pull --ff-only || log "WARNING: git pull failed, using existing data"
elif [[ -n "${GIT_SSH_KEY:-}" ]]; then
    log "Cloning training data from $DATA_REPO_URL..."
    mkdir -p ~/.ssh
    echo "$GIT_SSH_KEY" > ~/.ssh/id_ed25519
    chmod 600 ~/.ssh/id_ed25519
    ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null

    git clone --depth 1 "$DATA_REPO_URL" "$DATA_DIR" \
        && log "Training data cloned to $DATA_DIR" \
        || error_exit "Failed to clone training data repo"

    rm -f ~/.ssh/id_ed25519
    log "SSH key cleaned up."
else
    log "WARNING: No GIT_SSH_KEY set and no cached data found. Training data unavailable."
fi

# ---------------------------------------------------------------------------
# TensorBoard (optional)
# ---------------------------------------------------------------------------
if [[ "${START_TENSORBOARD:-false}" == "true" ]]; then
    log "Starting TensorBoard on port 6006..."
    if command -v tensorboard &>/dev/null; then
        tensorboard \
            --logdir /workspace/outputs/logs/"$EXPERIMENT" \
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
# SSH server (enables RunPod web terminal and SSH access)
# ---------------------------------------------------------------------------
if command -v sshd &>/dev/null; then
    mkdir -p /var/run/sshd
    # Allow root login for RunPod web terminal
    sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config 2>/dev/null || true
    sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config 2>/dev/null || true
    # Set a default root password if PUBLIC_KEY is not set
    if [[ -z "${PUBLIC_KEY:-}" ]]; then
        echo "root:runpod" | chpasswd 2>/dev/null || true
    fi
    /usr/sbin/sshd 2>/dev/null || log "WARNING: Failed to start sshd"
    log "SSH server started on port 22."
fi

# ---------------------------------------------------------------------------
# Execute CMD, then optionally keep alive for interactive access
# ---------------------------------------------------------------------------
if [[ $# -gt 0 ]]; then
    log "Executing CMD: $*"
    "$@"
    CMD_EXIT=$?
    log "CMD exited with code $CMD_EXIT"
else
    log "No CMD provided."
    CMD_EXIT=0
fi

# Keep the pod alive for interactive debugging/inspection
if [[ "${KEEP_ALIVE:-true}" == "true" ]]; then
    log "KEEP_ALIVE=true — pod will stay running. Connect via RunPod web terminal or SSH."
    log "To stop the pod, terminate it from the RunPod dashboard."
    sleep infinity
else
    exit $CMD_EXIT
fi
