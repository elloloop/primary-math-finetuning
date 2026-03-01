# Qwen2.5-7B Math Fine-tuning System

This repository provides a structured training and evaluation pipeline for fine-tuning `Qwen/Qwen2.5-7B-Instruct` on grade-school math tasks with LoRA.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_data.py --num_samples 200 --output data/examples/train.json --validate
python scripts/train.py --data_path data/examples/train.json --quick_test
python scripts/evaluate.py --model_path outputs/models/default/final --benchmark gsm8k
```

## Docker Images

This project builds 3 Docker images on each GitHub release:

| Image | Purpose |
|-------|---------|
| `ghcr.io/elloloop/primary-math-finetuning/train` | Fine-tune Qwen2.5-7B with LoRA on RunPod GPU Pods |
| `ghcr.io/elloloop/primary-math-finetuning/inference` | Serve the model via FastAPI + vLLM (RunPod Serverless) |
| `ghcr.io/elloloop/primary-math-finetuning/eval` | Benchmark against GSM8K on RunPod GPU Pods |

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for the full RunPod walkthrough.

## Docs
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — RunPod deployment guide (start here)
- [`docs/TRAINING.md`](docs/TRAINING.md)
- [`docs/EVALUATION.md`](docs/EVALUATION.md)
- [`docs/DATA_PREPARATION.md`](docs/DATA_PREPARATION.md)
- [`docs/API.md`](docs/API.md)
- [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)
