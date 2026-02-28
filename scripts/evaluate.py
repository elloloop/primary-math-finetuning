#!/usr/bin/env python
import argparse
from pathlib import Path
from src.evaluation.benchmark_runner import run_gsm8k_stub


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--benchmark", default="gsm8k", choices=["gsm8k", "math", "svamp", "asdiv", "all"])
    p.add_argument("--num_samples", type=int)
    p.add_argument("--num_fewshot", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--output_dir", default="./outputs/results")
    p.add_argument("--use_cot", action="store_true")
    p.add_argument("--save_predictions", action="store_true")
    p.add_argument("--error_analysis", action="store_true")
    p.add_argument("--full", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    res = run_gsm8k_stub(args.output_dir)
    print(f"{res['benchmark']} accuracy: {res['accuracy']:.2f}%")


if __name__ == "__main__":
    main()
