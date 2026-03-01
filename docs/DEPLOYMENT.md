# Deployment Guide

This project ships 3 Docker images, each built automatically on GitHub release:

| Image | Purpose | RunPod Target |
|-------|---------|---------------|
| `ghcr.io/elloloop/primary-math-finetuning/train:<version>` | Fine-tune Qwen2.5-7B with LoRA | GPU Pod |
| `ghcr.io/elloloop/primary-math-finetuning/inference:<version>` | Serve the model via FastAPI + vLLM | Serverless Endpoint |
| `ghcr.io/elloloop/primary-math-finetuning/eval:<version>` | Benchmark against GSM8K | GPU Pod |

## Prerequisites

- A [RunPod](https://www.runpod.io/) account with GPU credits
- A [HuggingFace](https://huggingface.co/) account with an access token (for gated models)
- A GitHub release tagged (e.g., `v1.0.0`) so the images are published to GHCR

## Overview

```
┌─────────────────────────────────────────────────────────┐
│                   RunPod Network Volume                  │
│                  /workspace/outputs/models               │
│                                                         │
│    ┌──────────┐     ┌────────────┐     ┌──────────┐    │
│    │  Train   │────>│  Inference  │<────│   Eval   │    │
│    │ GPU Pod  │     │ Serverless  │     │ GPU Pod  │    │
│    └──────────┘     └────────────┘     └──────────┘    │
│     writes model     reads model       reads model /    │
│     artifacts        serves on :8000   calls inference   │
└─────────────────────────────────────────────────────────┘
```

---

## Step 1: Create a RunPod Network Volume

Network Volumes persist data across pods and allow the train, inference, and eval containers to share model artifacts.

1. Go to **RunPod Console > Storage > Network Volumes**
2. Click **Create Network Volume**
3. Choose:
   - **Name**: `math-finetuning`
   - **Region**: same region you'll use for pods (e.g., `US-TX-3`)
   - **Size**: 50 GB (LoRA adapters are small, but base model cache needs space)
4. Note the volume ID — you'll attach it to pods as `/workspace`

---

## Step 2: Upload Training Data

The train image includes small example data at `data/examples/train.json`, but for real training you need to provide your own dataset. Three options:

### Option A: Upload to Network Volume first (recommended)

1. Create a lightweight **CPU pod** (cheapest available) and attach the `math-finetuning` network volume
2. Use the RunPod web terminal or `rsync`/`scp` to upload your JSON/JSONL training data to `/workspace/data/`
3. Terminate the upload pod
4. Now launch the train pod with the same volume — your data is at `/workspace/data/`

### Option B: Use HuggingFace datasets

Set `--data_path` to a HuggingFace dataset identifier (e.g., `gsm8k`). No volume is needed for data input (only for outputs).

```bash
python scripts/train.py --data_path gsm8k --output_dir /workspace/outputs/models/run1
```

### Option C: Bake data into a custom image

For reproducible runs, extend the train image with your data:

```dockerfile
FROM ghcr.io/elloloop/primary-math-finetuning/train:latest
COPY my_data/ /workspace/data/
```

---

## Step 3: Train (GPU Pod)

### Create a GPU Pod Template

1. Go to **RunPod Console > Pods > Templates > New Template**
2. Configure:
   - **Template Name**: `math-finetune-train`
   - **Container Image**: `ghcr.io/elloloop/primary-math-finetuning/train:latest`
   - **Container Disk**: 20 GB
   - **Volume Mount Path**: `/workspace`
   - **Environment Variables**:
     ```
     HF_TOKEN=hf_your_token_here
     WANDB_API_KEY=your_wandb_key       # optional, for experiment tracking
     START_TENSORBOARD=true              # optional, launches TensorBoard on :6006
     ```
   - **Expose Ports**: `6006` (TensorBoard, optional)

### Launch Training

1. Go to **Pods > Deploy** and select your template
2. Choose a GPU — **A40 (48 GB)** or **A100 (80 GB)** recommended
3. Attach the `math-finetuning` network volume
4. Click **Deploy**

The container will:
- Run `runpod_startup.sh` (GPU check, directory setup)
- Execute `scripts/quick_test.py` by default (verifies the setup works)

To run full training, override the command:

```bash
# Connect via RunPod web terminal or SSH, then:
python scripts/train.py \
    --data_path data/examples/train.json \
    --output_dir /workspace/outputs/models/run1 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4
```

Or set the **Docker Command** in the template to run training automatically:

```
python scripts/train.py --data_path /workspace/data/train.json --output_dir /workspace/outputs/models/run1
```

#### Train + Eval in one pod

The train image includes the GSM8K eval scripts. To run evaluation immediately after training (avoiding a separate eval pod), chain the commands:

```
bash -c "python scripts/train.py --data_path /workspace/data/train.json --output_dir /workspace/outputs/models/run1 && MODEL_PATH=/workspace/outputs/models/run1/final python eval/run_benchmarks.py"
```

Set this as the **Docker Command** in your pod template. If training fails, eval is skipped automatically.


### Training Output

After training completes, the network volume will contain:

```
/workspace/outputs/models/run1/
├── final/              # merged model (or LoRA adapter)
├── checkpoints/        # intermediate checkpoints
└── logs/               # TensorBoard logs
```

### Terminate the Pod

Once training is done, **stop or terminate** the pod to stop billing. The network volume retains all data.

---

## Step 4: Serve Inference (RunPod Serverless)

RunPod Serverless auto-scales GPU workers based on request volume — ideal for serving the model in production.

### Create a Serverless Endpoint

1. Go to **RunPod Console > Serverless > New Endpoint**
2. Configure:
   - **Endpoint Name**: `math-inference`
   - **Container Image**: `ghcr.io/elloloop/primary-math-finetuning/inference:latest`
   - **GPU Type**: A40 or A100
   - **Min Workers**: 0 (scales to zero when idle)
   - **Max Workers**: 3 (adjust based on traffic)
   - **Network Volume**: attach `math-finetuning` volume
   - **Environment Variables**:
     ```
     MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
     LORA_PATH=/workspace/outputs/models/run1/final
     MAX_MODEL_LEN=4096
     GPU_MEMORY_UTILIZATION=0.9
     ```
3. Click **Create**

> **Note:** If you don't have a network volume attached, set `MODEL_PATH` to a HuggingFace Hub repo and omit `LORA_PATH`. The model will be downloaded on cold start.

### API Usage

Once the endpoint is active, you get a RunPod endpoint URL. The inference server exposes:

#### Health Check
```bash
curl https://api.runpod.ai/v2/<endpoint_id>/health
```

#### Single Completion
```bash
curl -X POST https://api.runpod.ai/v2/<endpoint_id>/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "Solve this math problem step by step:\nSally has 3 apples. She buys 5 more. How many does she have?\nAnswer:",
      "max_tokens": 512,
      "temperature": 0.0
    }
  }'
```

#### Batch Completions
```bash
curl -X POST https://api.runpod.ai/v2/<endpoint_id>/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompts": ["Problem 1...", "Problem 2..."],
      "max_tokens": 512,
      "temperature": 0.0
    }
  }'
```

### Alternative: Serve on a GPU Pod

If you don't need auto-scaling, you can run inference on a regular GPU Pod:

1. Deploy a pod with the `inference` image
2. Expose port **8000**
3. Access directly at `http://<pod_ip>:8000`

```bash
# Health check
curl http://<pod_ip>:8000/health

# Completion
curl -X POST http://<pod_ip>:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Solve: 3 + 5 = ?", "max_tokens": 256}'
```

---

## Step 5: Evaluate (GPU Pod, optional)

> **Tip:** If you used the train + eval chaining command in Step 3, eval already ran on the same pod and you can skip this step. The standalone eval image is useful when you want to re-evaluate a model without retraining.

### Option A: Direct Model Loading

Loads the model on the eval pod's GPU and runs benchmarks locally.

1. Deploy a GPU Pod with the `eval` image
2. Attach the `math-finetuning` network volume
3. Set environment variables:
   ```
   MODEL_PATH=/workspace/outputs/models/run1/final
   OUTPUT_DIR=/workspace/outputs/results
   NUM_SAMPLES=200          # omit for full test set
   BATCH_SIZE=8
   ```
4. The container runs `eval/run_benchmarks.py` automatically

### Option B: Remote Inference

Sends prompts to the running inference server — useful when you want to eval without loading the model again.

1. Deploy a **CPU Pod** (no GPU needed) with the `eval` image
2. Set environment variables:
   ```
   INFERENCE_URL=http://<inference_pod_ip>:8000
   OUTPUT_DIR=/workspace/outputs/results
   NUM_SAMPLES=200
   ```

### Eval Output

Results are saved to the output directory:

```
/workspace/outputs/results/benchmark_results.json
```

```json
{
  "model_path": "/workspace/outputs/models/run1/final",
  "elapsed_seconds": 342.5,
  "benchmarks": {
    "gsm8k": {
      "accuracy": 0.6850,
      "total": 200,
      "correct": 137,
      "incorrect": 63
    }
  }
}
```

The container also prints a summary to stdout:

```
============================================================
BENCHMARK RESULTS
============================================================
  gsm8k:
    Accuracy:  0.6850
    Correct:   137 / 200
  Time: 342.5s
============================================================
```

---

## End-to-End Workflow Summary

```bash
# 1. Create a network volume in RunPod Console (50 GB, same region as pods)

# 2. Upload training data — launch a cheap CPU pod, upload to /workspace/data/, terminate

# 3. Train + Eval — launch a GPU Pod
#    Image: ghcr.io/elloloop/primary-math-finetuning/train:1.0.0
#    GPU: A40 or A100
#    Volume: math-finetuning -> /workspace
#    Command: bash -c "python scripts/train.py --data_path /workspace/data/train.json \
#      --output_dir /workspace/outputs/models/run1 \
#      && MODEL_PATH=/workspace/outputs/models/run1/final python eval/run_benchmarks.py"

# 4. Serve — create a Serverless Endpoint
#    Image: ghcr.io/elloloop/primary-math-finetuning/inference:1.0.0
#    Volume: math-finetuning -> /workspace
#    Env: MODEL_PATH=Qwen/Qwen2.5-7B-Instruct LORA_PATH=/workspace/outputs/models/run1/final


# 5. Check results
#    cat /workspace/outputs/results/benchmark_results.json

# 6. Terminate pods when done (network volume persists)
```

---

## Environment Variable Reference

### Train Image

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | HuggingFace token for gated model access |
| `WANDB_API_KEY` | — | Weights & Biases API key (optional) |
| `START_TENSORBOARD` | `false` | Launch TensorBoard on port 6006 |
| `TRANSFORMERS_CACHE` | `/workspace/cache` | Model download cache directory |

### Inference Image

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `Qwen/Qwen2.5-7B` | Model path (local dir or HF Hub ID) |
| `LORA_PATH` | — | Path to LoRA adapter directory |
| `MAX_MODEL_LEN` | `4096` | Maximum sequence length |
| `GPU_MEMORY_UTILIZATION` | `0.9` | Fraction of GPU memory to use |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |

### Eval Image

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | — | Model path for local loading |
| `LORA_PATH` | — | LoRA adapter path (local mode only) |
| `INFERENCE_URL` | — | Inference server URL (remote mode) |
| `OUTPUT_DIR` | `/workspace/outputs/results` | Results output directory |
| `NUM_SAMPLES` | all | Limit eval to N samples |
| `BATCH_SIZE` | `8` | Batch size for generation |

---

## Troubleshooting

**Pod can't pull the image**
- Ensure the GHCR packages are set to public (the release workflow does this automatically)
- Or log in to GHCR on the pod: `docker login ghcr.io -u <github_user> -p <github_pat>`

**Out of GPU memory during training**
- Reduce `per_device_train_batch_size` (try 2 or 1)
- Enable gradient checkpointing in the training config
- Use an A100 (80 GB) instead of A40 (48 GB)

**Inference server won't start**
- Check `MODEL_PATH` is correct and accessible
- Reduce `GPU_MEMORY_UTILIZATION` to `0.8`
- Reduce `MAX_MODEL_LEN` to `2048`

**Network volume not visible**
- Ensure the pod and volume are in the **same region**
- Check the volume is mounted at `/workspace` in the pod template
