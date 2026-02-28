#!/usr/bin/env python
"""Evaluation CLI for primary-math fine-tuned models.

Runs one or more benchmarks against a model checkpoint and prints results.
Optionally writes a machine-readable JSON report for CI integration.
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
from src.evaluation.analytics import (
    generate_error_report,
    print_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a fine-tuned model on math benchmarks.",
    )
    p.add_argument(
        "--model_path",
        required=True,
        help="Path to model directory or HuggingFace Hub identifier.",
    )
    p.add_argument(
        "--benchmark",
        default="gsm8k",
        choices=["gsm8k", "all"],
        help="Which benchmark(s) to run.",
    )
    p.add_argument(
        "--num_samples", type=int, default=None, help="Limit evaluation to N samples."
    )
    p.add_argument(
        "--num_fewshot", type=int, default=5, help="Number of few-shot exemplars."
    )
    p.add_argument("--batch_size", type=int, default=8, help="Generation batch size.")
    p.add_argument(
        "--output_dir", default="./outputs/results", help="Directory for result files."
    )
    p.add_argument(
        "--use_cot", action="store_true", help="Use chain-of-thought prompting."
    )
    p.add_argument(
        "--save_predictions", action="store_true", help="Save per-example predictions."
    )
    p.add_argument(
        "--error_analysis", action="store_true", help="Run error analysis on results."
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Run on the full test set (overrides --num_samples).",
    )
    p.add_argument(
        "--output-json", default=None, help="Path for CI-friendly JSON output."
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_samples = None if args.full else args.num_samples

    # ---- Load model via BenchmarkRunner ---------------------------------
    logger.info("Loading model from %s", args.model_path)
    try:
        runner = BenchmarkRunner(model_path=args.model_path)
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        return 1

    # ---- Run benchmark(s) -----------------------------------------------
    all_results: dict = {}

    if args.benchmark in ("gsm8k", "all"):
        logger.info(
            "Running GSM8K evaluation (num_samples=%s, num_fewshot=%d)",
            num_samples,
            args.num_fewshot,
        )
        gsm8k_results = runner.run_gsm8k(
            num_samples=num_samples,
            num_fewshot=args.num_fewshot,
        )
        all_results["gsm8k"] = gsm8k_results

        print_summary(gsm8k_results)

        # Save predictions if requested
        if args.save_predictions:
            preds_path = output_dir / "gsm8k_predictions.json"
            with open(preds_path, "w", encoding="utf-8") as f:
                json.dump(
                    gsm8k_results.get("predictions", []), f, indent=2, default=str
                )
            logger.info("Predictions saved to %s", preds_path)

        # Error analysis
        if args.error_analysis:
            report = generate_error_report(
                gsm8k_results, output_path=str(output_dir / "gsm8k_error_report.json")
            )
            error_info = report.get("error_analysis", {})
            print("\n--- Error Analysis ---")
            print(f"  Total errors:   {error_info.get('total_errors', 0)}")
            print(f"  Error rate:     {error_info.get('error_rate', 0):.4f}")
            by_type = error_info.get("by_error_type", {})
            if by_type:
                print("  By error type:")
                for etype, count in by_type.items():
                    print(f"    {etype}: {count}")
            print()

    # Save combined results
    runner.save_results(
        {"model_path": args.model_path, "benchmarks": all_results},
        str(output_dir),
    )

    # ---- CI JSON output -------------------------------------------------
    if args.output_json:
        ci_output: dict = {"model_path": args.model_path, "benchmarks": {}}
        for bench_name, bench_results in all_results.items():
            ci_output["benchmarks"][bench_name] = {
                "accuracy": bench_results.get("overall_accuracy", 0.0),
                "total": bench_results.get("total_problems", 0),
                "correct": bench_results.get("correct", 0),
                "incorrect": bench_results.get("incorrect", 0),
            }
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(ci_output, f, indent=2)
        logger.info("CI JSON output written to %s", json_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
