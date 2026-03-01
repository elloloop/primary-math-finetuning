#!/usr/bin/env python
"""Batch evaluation entrypoint for benchmarking fine-tuned models.

Supports two modes:
  1. Local model loading (MODEL_PATH) — loads the model directly for evaluation.
  2. Remote inference (INFERENCE_URL) — sends prompts to the inference server.

Environment variables:
  MODEL_PATH     — path to a local model or HuggingFace Hub identifier
  LORA_PATH      — optional path to a LoRA adapter (local mode only)
  INFERENCE_URL  — URL of the inference server (e.g. http://inference:8000)
  OUTPUT_DIR     — directory for result files (default: /workspace/outputs/results)
  NUM_SAMPLES    — limit evaluation to N samples (default: all)
  BATCH_SIZE     — generation batch size (default: 8)
"""

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

_GSM8K_ANSWER_RE = re.compile(r"####\s*(-?\d[\d,]*\.?\d*)")


def _normalize_numeric(s: str) -> str:
    """Strip commas and trailing '.0' from a numeric string."""
    s = s.replace(",", "").strip()
    try:
        n = float(s)
        if n == int(n):
            return str(int(n))
        return str(n)
    except ValueError:
        return s


def _extract_numeric_answer(text: str) -> str:
    """Extract the final numeric answer after #### from *text*.

    Uses the same regex as the local GSM8K evaluator so that scoring is
    consistent across local and remote evaluation paths.
    """
    match = _GSM8K_ANSWER_RE.search(text)
    if match is None:
        return ""
    return _normalize_numeric(match.group(1))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "")
LORA_PATH = os.environ.get("LORA_PATH", "")
INFERENCE_URL = os.environ.get("INFERENCE_URL", "")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/workspace/outputs/results")
NUM_SAMPLES = os.environ.get("NUM_SAMPLES", "")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))


def run_local_eval(model_path: str, lora_path: str, num_samples: int | None) -> dict:
    """Run evaluation by loading the model locally."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.evaluation.benchmark_runner import BenchmarkRunner
    from src.evaluation.analytics import print_summary

    runner = BenchmarkRunner(model_path=model_path, lora_path=lora_path)
    logger.info("Running GSM8K evaluation (num_samples=%s)", num_samples)
    results = runner.run_gsm8k(num_samples=num_samples)
    print_summary(results)
    return results


def run_remote_eval(inference_url: str, num_samples: int | None) -> dict:
    """Run evaluation by calling the remote inference server."""
    import requests
    from datasets import load_dataset

    logger.info("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    total = len(dataset)
    correct = 0
    predictions = []

    # Process in batches
    for i in range(0, total, BATCH_SIZE):
        batch = dataset[i : i + BATCH_SIZE]
        prompts = [
            f"Solve this math problem step by step:\n{q}\nAnswer:"
            for q in batch["question"]
        ]

        resp = requests.post(
            f"{inference_url.rstrip('/')}/v1/batch",
            json={"prompts": prompts, "max_tokens": 512, "temperature": 0.0},
            timeout=300,
        )
        resp.raise_for_status()
        completions = resp.json()["completions"]

        for j, completion in enumerate(completions):
            answer_text = batch["answer"][j]
            # Extract the final numeric answer after ####
            expected = _extract_numeric_answer(answer_text)
            generated = completion["text"]

            # Extract numeric answer using the same #### pattern as the local evaluator
            predicted = _extract_numeric_answer(generated)
            is_correct = expected != "" and predicted == expected
            correct += int(is_correct)
            predictions.append(
                {
                    "question": batch["question"][j],
                    "expected": expected,
                    "generated": generated,
                    "correct": is_correct,
                }
            )

        logger.info("Processed %d/%d examples", min(i + BATCH_SIZE, total), total)

    accuracy = correct / total if total > 0 else 0.0
    return {
        "benchmark": "gsm8k",
        "overall_accuracy": accuracy,
        "total_problems": total,
        "correct": correct,
        "incorrect": total - correct,
        "predictions": predictions,
    }


def main() -> int:
    if not MODEL_PATH and not INFERENCE_URL:
        logger.error("Set MODEL_PATH or INFERENCE_URL environment variable")
        return 1

    num_samples = int(NUM_SAMPLES) if NUM_SAMPLES else None
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    if INFERENCE_URL:
        logger.info("Using remote inference at %s", INFERENCE_URL)
        results = run_remote_eval(INFERENCE_URL, num_samples)
    else:
        logger.info("Loading model locally from %s", MODEL_PATH)
        results = run_local_eval(MODEL_PATH, LORA_PATH, num_samples)

    elapsed = time.time() - start_time

    # Build summary report
    report = {
        "model_path": MODEL_PATH or f"remote:{INFERENCE_URL}",
        "elapsed_seconds": round(elapsed, 2),
        "benchmarks": {
            "gsm8k": {
                "accuracy": results.get("overall_accuracy", 0.0),
                "total": results.get("total_problems", 0),
                "correct": results.get("correct", 0),
                "incorrect": results.get("incorrect", 0),
            }
        },
    }

    # Save results
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Results saved to %s", results_path)

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    for name, bench in report["benchmarks"].items():
        print(f"  {name}:")
        print(f"    Accuracy:  {bench['accuracy']:.4f}")
        print(f"    Correct:   {bench['correct']} / {bench['total']}")
    print(f"  Time: {elapsed:.1f}s")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
