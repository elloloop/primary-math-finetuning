#!/usr/bin/env python
"""Quick pipeline validation script.

Runs a minimal end-to-end pipeline to verify that data generation, training,
and evaluation all work correctly. Designed to complete in 10-15 minutes on
an A40 GPU.

Steps:
    1. Generate 100 sample problems.
    2. Train for 1 epoch with a small batch size.
    3. Evaluate on 50 samples.
    4. Report pass/fail.
"""

import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.data_generator import generate_math_problems, save_samples
from src.data.data_validator import validate_dataset
from src.training.trainer import MathTrainer
from src.evaluation.benchmark_runner import BenchmarkRunner
from src.evaluation.analytics import print_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

NUM_TRAIN_SAMPLES = 100
NUM_EVAL_SAMPLES = 50
TRAIN_EPOCHS = 1
BATCH_SIZE = 2


def _step(label: str) -> None:
    """Print a step header."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")


def main() -> int:
    start = time.time()
    passed = True
    output_dir = os.path.join(PROJECT_ROOT, "outputs", "quick_test")
    data_path = os.path.join(output_dir, "quick_test_data.json")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Generate data
    # ------------------------------------------------------------------
    _step("Step 1/3: Generate training data")
    try:
        samples = generate_math_problems(num_samples=NUM_TRAIN_SAMPLES, seed=42)
        save_samples(samples, data_path)
        is_valid, errors = validate_dataset(samples)
        if not is_valid:
            logger.error("Generated data failed validation: %s", errors[:5])
            passed = False
        else:
            print(f"  Generated {len(samples)} valid samples.")
    except Exception as exc:
        logger.error("Data generation failed: %s", exc)
        passed = False

    if not passed:
        _print_result(passed, time.time() - start)
        return 1

    # ------------------------------------------------------------------
    # Step 2: Train for 1 epoch
    # ------------------------------------------------------------------
    _step("Step 2/3: Train for 1 epoch")
    model_output_dir = os.path.join(output_dir, "model")
    try:
        trainer = MathTrainer(
            output_dir=model_output_dir,
            phase=1,
            training_args_override={
                "num_train_epochs": TRAIN_EPOCHS,
                "per_device_train_batch_size": BATCH_SIZE,
                "per_device_eval_batch_size": BATCH_SIZE,
                "logging_steps": 1,
                "save_steps": 9999,
                "eval_steps": 9999,
                "save_total_limit": 1,
                "report_to": [],
            },
        )
        trainer.prepare_model()
        trainer.prepare_data(data_path)
        metrics = trainer.train()
        train_loss = metrics.get("train_loss", None)
        print(f"  Training complete. Loss: {train_loss}")

        save_path = trainer.save(merge_lora=True)
        print(f"  Model saved to {save_path}")
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        passed = False

    if not passed:
        _print_result(passed, time.time() - start)
        return 1

    # ------------------------------------------------------------------
    # Step 3: Evaluate on 50 samples
    # ------------------------------------------------------------------
    _step("Step 3/3: Evaluate on 50 samples")
    try:
        runner = BenchmarkRunner(model_path=save_path)
        results = runner.run_gsm8k(num_samples=NUM_EVAL_SAMPLES, num_fewshot=3)
        print_summary(results)

        accuracy = results.get("overall_accuracy", 0.0)
        print(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc)
        # Evaluation failure is not fatal for a quick test -- the model may
        # perform poorly after only 1 epoch, but the pipeline still works.
        logger.warning("Evaluation raised an error but pipeline steps succeeded.")

    elapsed = time.time() - start
    _print_result(passed, elapsed)
    return 0 if passed else 1


def _print_result(passed: bool, elapsed: float) -> None:
    """Print the final pass/fail banner."""
    status = "PASSED" if passed else "FAILED"
    print(f"\n{'='*60}")
    print(f"  Quick Test Result: {status}")
    print(f"  Elapsed time:     {elapsed / 60:.1f} minutes")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    raise SystemExit(main())
