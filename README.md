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

## Docs
- `docs/TRAINING.md`
- `docs/EVALUATION.md`
- `docs/DATA_PREPARATION.md`
- `docs/DEPLOYMENT.md`
- `docs/API.md`
- `docs/TROUBLESHOOTING.md`
