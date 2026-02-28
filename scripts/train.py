#!/usr/bin/env python
"""Main training CLI for primary-math fine-tuning.

Exit codes:
    0 - Success
    1 - General error
    2 - Configuration error
    3 - Data error
    4 - CUDA / device error
    5 - Out of memory
"""

import argparse
import logging
import os
import sys
import time

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that `src.*` and `config.*`
# imports work regardless of how the script is invoked.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.training_config import PHASES
from src.training.trainer import MathTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune a causal LM on primary-math data using LoRA.",
    )
    p.add_argument(
        "--data_path",
        required=True,
        help="Path to training data (JSON/JSONL file or HuggingFace dataset name).",
    )
    p.add_argument(
        "--output_dir",
        default="./outputs/models/default",
        help="Directory for checkpoints, logs, and saved models.",
    )
    p.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help="Training phase (1-4). Each phase uses a different LR / epoch schedule.",
    )
    p.add_argument("--num_epochs", type=int, default=None, help="Override number of training epochs.")
    p.add_argument("--batch_size", type=int, default=None, help="Per-device train and eval batch size.")
    p.add_argument("--learning_rate", type=float, default=None, help="Override learning rate.")
    p.add_argument("--lora_r", type=int, default=None, help="LoRA rank (r).")
    p.add_argument("--max_seq_length", type=int, default=None, help="Maximum sequence length.")
    p.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging.")
    p.add_argument("--resume_from_checkpoint", default=None, help="Path to checkpoint directory to resume from.")
    p.add_argument("--quick_test", action="store_true", help="Use a tiny subset for a fast sanity check.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    phase_info = PHASES.get(args.phase, {})
    phase_name = phase_info.get("name", "unknown")
    logger.info("Starting training -- phase %d (%s)", args.phase, phase_name)

    # ---- Build overrides ------------------------------------------------
    training_overrides: dict = {}
    if args.num_epochs is not None:
        training_overrides["num_train_epochs"] = args.num_epochs
    if args.batch_size is not None:
        training_overrides["per_device_train_batch_size"] = args.batch_size
        training_overrides["per_device_eval_batch_size"] = args.batch_size
    if args.learning_rate is not None:
        training_overrides["learning_rate"] = args.learning_rate
    if args.max_seq_length is not None:
        training_overrides["max_seq_length"] = args.max_seq_length

    report_to = ["tensorboard"]
    if args.use_wandb:
        report_to.append("wandb")
    training_overrides["report_to"] = report_to

    if args.quick_test:
        training_overrides.setdefault("num_train_epochs", 1)
        training_overrides.setdefault("per_device_train_batch_size", 2)
        training_overrides.setdefault("per_device_eval_batch_size", 2)
        training_overrides["logging_steps"] = 1
        training_overrides["save_steps"] = 50
        training_overrides["eval_steps"] = 50

    lora_overrides: dict = {}
    if args.lora_r is not None:
        lora_overrides["r"] = args.lora_r

    # ---- Instantiate trainer --------------------------------------------
    try:
        trainer = MathTrainer(
            output_dir=args.output_dir,
            phase=args.phase,
            training_args_override=training_overrides,
            lora_config_override=lora_overrides,
        )
    except Exception as exc:
        logger.error("Configuration error: %s", exc)
        return 2

    # ---- Prepare model --------------------------------------------------
    try:
        trainer.prepare_model()
    except torch.cuda.OutOfMemoryError:
        return 5
    except RuntimeError as exc:
        if "CUDA" in str(exc) or "cuda" in str(exc):
            logger.error("CUDA error during model preparation: %s", exc)
            return 4
        logger.error("Runtime error during model preparation: %s", exc)
        return 1
    except Exception as exc:
        logger.error("Failed to prepare model: %s", exc)
        return 1

    # ---- Prepare data ---------------------------------------------------
    try:
        trainer.prepare_data(args.data_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Data error: %s", exc)
        return 3
    except Exception as exc:
        logger.error("Failed to prepare data: %s", exc)
        return 3

    # ---- Train ----------------------------------------------------------
    start_time = time.time()
    try:
        metrics = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    except torch.cuda.OutOfMemoryError:
        return 5
    except RuntimeError as exc:
        if "CUDA out of memory" in str(exc) or "OutOfMemoryError" in str(exc):
            return 5
        if "CUDA" in str(exc) or "cuda" in str(exc):
            logger.error("CUDA error during training: %s", exc)
            return 4
        logger.error("Runtime error during training: %s", exc)
        return 1
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        return 1

    elapsed = time.time() - start_time

    # ---- Evaluate -------------------------------------------------------
    eval_metrics = {}
    try:
        eval_metrics = trainer.evaluate()
    except RuntimeError:
        logger.warning("Evaluation skipped (no eval dataset or trainer not ready).")

    # ---- Save -----------------------------------------------------------
    try:
        save_path = trainer.save(merge_lora=True)
    except Exception as exc:
        logger.error("Failed to save model: %s", exc)
        return 1

    # ---- Summary --------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Training Summary")
    print("=" * 60)
    print(f"  Phase:            {args.phase} ({phase_name})")
    print(f"  Data path:        {args.data_path}")
    print(f"  Output dir:       {args.output_dir}")
    print(f"  Wall-clock time:  {elapsed / 60:.1f} minutes")
    if metrics:
        train_loss = metrics.get("train_loss", metrics.get("loss", "N/A"))
        print(f"  Train loss:       {train_loss}")
    if eval_metrics:
        eval_loss = eval_metrics.get("eval_loss", "N/A")
        print(f"  Eval loss:        {eval_loss}")
    print(f"  Saved model:      {save_path}")
    print("=" * 60 + "\n")

    return 0


# We need torch for OOM guards; import at module level after sys.path setup.
import torch  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
