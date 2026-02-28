#!/usr/bin/env python
"""Compare multiple model checkpoints on a quick evaluation.

Takes a list of checkpoint paths, runs a short GSM8K evaluation on each,
and prints a side-by-side comparison table.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.benchmark_runner import BenchmarkRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare multiple model checkpoints on GSM8K.",
    )
    p.add_argument(
        "checkpoints",
        nargs="+",
        help="Paths to model checkpoint directories.",
    )
    p.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of GSM8K samples to evaluate per checkpoint.",
    )
    p.add_argument(
        "--num_fewshot",
        type=int,
        default=5,
        help="Number of few-shot exemplars.",
    )
    p.add_argument(
        "--output_json",
        default=None,
        help="Optional path to save comparison results as JSON.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    results_table: list[dict] = []

    for ckpt_path in args.checkpoints:
        ckpt_name = os.path.basename(ckpt_path.rstrip("/")) or ckpt_path
        logger.info("Evaluating checkpoint: %s", ckpt_path)

        try:
            runner = BenchmarkRunner(model_path=ckpt_path)
            gsm8k_results = runner.run_gsm8k(
                num_samples=args.num_samples,
                num_fewshot=args.num_fewshot,
            )
            results_table.append(
                {
                    "checkpoint": ckpt_name,
                    "path": ckpt_path,
                    "accuracy": gsm8k_results["overall_accuracy"],
                    "correct": gsm8k_results["correct"],
                    "total": gsm8k_results["total_problems"],
                    "status": "ok",
                }
            )
        except Exception as exc:
            logger.error("Failed to evaluate %s: %s", ckpt_path, exc)
            results_table.append(
                {
                    "checkpoint": ckpt_name,
                    "path": ckpt_path,
                    "accuracy": None,
                    "correct": None,
                    "total": None,
                    "status": f"error: {exc}",
                }
            )

        # Free GPU memory between checkpoints
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        # Delete runner to release model from memory
        del runner

    # ---- Print comparison table -----------------------------------------
    print("\n" + "=" * 78)
    print("  Checkpoint Comparison")
    print("=" * 78)
    header = f"  {'Checkpoint':<30s} {'Accuracy':>10s} {'Correct':>10s} {'Total':>8s} {'Status':>10s}"
    print(header)
    print("  " + "-" * 74)

    for row in results_table:
        if row["accuracy"] is not None:
            acc_str = f"{row['accuracy']:.4f}"
            correct_str = str(row["correct"])
            total_str = str(row["total"])
        else:
            acc_str = "N/A"
            correct_str = "N/A"
            total_str = "N/A"
        print(
            f"  {row['checkpoint']:<30s} {acc_str:>10s} {correct_str:>10s} "
            f"{total_str:>8s} {row['status']:>10s}"
        )

    print("=" * 78 + "\n")

    # Identify best checkpoint
    valid = [r for r in results_table if r["accuracy"] is not None]
    if valid:
        best = max(valid, key=lambda r: r["accuracy"])
        print(
            f"  Best checkpoint: {best['checkpoint']} (accuracy={best['accuracy']:.4f})"
        )
        print()

    # ---- Optionally save JSON -------------------------------------------
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results_table, f, indent=2, default=str)
        logger.info("Comparison results saved to %s", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
