#!/usr/bin/env python
import argparse
import sys
from src.data.data_loader import load_training_dataset
from src.models.model_loader import load_model_and_tokenizer
from src.training.trainer import tokenize_dataset, run_training
from config.training_config import PHASES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_dir", default="./outputs/models/default")
    p.add_argument("--phase", type=int, choices=[1, 2, 3, 4])
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--resume_from_checkpoint")
    p.add_argument("--quick_test", action="store_true")
    p.add_argument("--use_wandb", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        ds = load_training_dataset(args.data_path, quick_test=args.quick_test)
        split = ds.train_test_split(test_size=0.1, seed=42)
        model, tokenizer = load_model_and_tokenizer(lora_r=args.lora_r)
        train_ds = tokenize_dataset(split["train"], tokenizer, args.max_seq_length)
        eval_ds = tokenize_dataset(split["test"], tokenizer, args.max_seq_length)

        overrides = {
            "num_train_epochs": args.num_epochs,
            "per_device_train_batch_size": args.batch_size,
            "per_device_eval_batch_size": args.batch_size,
            "learning_rate": PHASES.get(args.phase, {}).get("learning_rate", args.learning_rate),
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "report_to": ["tensorboard"] + (["wandb"] if args.use_wandb else []),
        }
        run_training(model, tokenizer, train_ds, eval_ds, args.output_dir, overrides)
        return 0
    except ValueError as e:
        print(e, file=sys.stderr)
        return 3
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("OOM: reduce batch size or max_seq_length", file=sys.stderr)
            return 5
        print(e, file=sys.stderr)
        return 4
    except Exception as e:
        print(e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
