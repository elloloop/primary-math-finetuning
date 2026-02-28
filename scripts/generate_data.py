#!/usr/bin/env python
"""Data generation CLI for primary-math training data.

Generates synthetic math word problems and optionally augments GSM8K samples.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.data_generator import generate_math_problems, augment_gsm8k, save_samples
from src.data.data_validator import validate_dataset, get_dataset_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic primary-math training data.",
    )
    p.add_argument(
        "--num_samples",
        type=int,
        required=True,
        help="Number of problems to generate.",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output file path (JSON).",
    )
    p.add_argument(
        "--difficulty",
        default="mixed",
        choices=["easy", "medium", "hard", "mixed"],
        help="Difficulty level for generated problems.",
    )
    p.add_argument(
        "--operations",
        default="mixed",
        choices=["addition", "subtraction", "multiplication", "division", "mixed"],
        help="Arithmetic operations to include.",
    )
    p.add_argument(
        "--augment_gsm8k",
        action="store_true",
        help="Also generate augmented samples from GSM8K training set.",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="Validate the generated dataset after saving.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Generate synthetic problems ------------------------------------
    logger.info(
        "Generating %d problems (difficulty=%s, operations=%s)",
        args.num_samples,
        args.difficulty,
        args.operations,
    )
    samples = generate_math_problems(
        num_samples=args.num_samples,
        difficulty=args.difficulty,
        operations=args.operations,
        seed=args.seed,
    )
    logger.info("Generated %d synthetic samples.", len(samples))

    # ---- Optionally augment with GSM8K ----------------------------------
    if args.augment_gsm8k:
        try:
            from src.data.data_loader import load_gsm8k

            logger.info("Loading GSM8K training set for augmentation...")
            gsm8k_ds = load_gsm8k(split="train")
            num_augmented = max(args.num_samples // 4, 50)
            augmented = augment_gsm8k(
                gsm8k_ds, num_augmented=num_augmented, seed=args.seed
            )
            samples.extend(augmented)
            logger.info(
                "Added %d augmented GSM8K samples (total: %d).",
                len(augmented),
                len(samples),
            )
        except Exception as exc:
            logger.warning(
                "GSM8K augmentation failed, continuing with synthetic data only: %s",
                exc,
            )

    # ---- Save -----------------------------------------------------------
    save_samples(samples, str(output_path))
    print(f"Saved {len(samples)} samples to {output_path}")

    # ---- Validate -------------------------------------------------------
    if args.validate:
        is_valid, errors = validate_dataset(samples)
        if is_valid:
            print("Validation: PASSED")
        else:
            print(f"Validation: FAILED ({len(errors)} errors)")
            for err in errors[:10]:
                print(f"  - {err}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")

        stats = get_dataset_stats(samples)
        print("\n--- Dataset Statistics ---")
        print(f"  Total samples:       {stats['total_samples']}")
        print(f"  By difficulty:       {stats['by_difficulty']}")
        print(f"  By category:         {stats['by_category']}")
        print(f"  By answer:           {stats['by_answer']}")
        print(f"  Avg question length: {stats['avg_question_length']}")
        print(f"  Has explanation:     {stats['has_explanation']}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
