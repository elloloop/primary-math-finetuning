"""Training utilities for metric computation, argument construction, and GPU diagnostics."""

import logging
import math
from typing import Any

import numpy as np
import torch
from transformers import DataCollatorForLanguageModeling, TrainingArguments

from config.training_config import PHASES, TRAINING_ARGS

logger = logging.getLogger(__name__)


def build_data_collator(tokenizer):
    """Build a causal-LM data collator (no masked-LM).

    Args:
        tokenizer: The tokenizer to use for padding.

    Returns:
        A DataCollatorForLanguageModeling instance with mlm=False.
    """
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


def compute_metrics(eval_pred) -> dict[str, float]:
    """Compute accuracy and perplexity from model evaluation predictions.

    Designed for use as the ``compute_metrics`` callback in a HuggingFace
    Trainer.  The logits are shifted by one position so that each token's
    prediction is compared against the next token in the sequence.

    Args:
        eval_pred: An ``EvalPrediction`` namedtuple with ``predictions``
            (logits array of shape [batch, seq_len, vocab]) and ``label_ids``
            (token-id array of shape [batch, seq_len]).

    Returns:
        Dictionary with keys ``accuracy`` and ``perplexity``.
    """
    logits, labels = eval_pred

    # Shift: predict next token
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    # Accuracy: compare argmax predictions to labels, ignoring padding (-100)
    predictions = np.argmax(shift_logits, axis=-1)
    mask = shift_labels != -100
    correct = (predictions == shift_labels) & mask
    accuracy = float(correct.sum()) / max(float(mask.sum()), 1.0)

    # Perplexity: exp(mean cross-entropy) over non-padding tokens
    # Use float32 to avoid overflow during softmax
    shift_logits_f32 = shift_logits.astype(np.float32)
    max_logits = np.max(shift_logits_f32, axis=-1, keepdims=True)
    exp_logits = np.exp(shift_logits_f32 - max_logits)
    log_sum_exp = np.log(np.sum(exp_logits, axis=-1)) + max_logits.squeeze(-1)

    # Gather the logits for the correct labels
    batch_size, seq_len = shift_labels.shape
    batch_idx = np.arange(batch_size)[:, None]
    seq_idx = np.arange(seq_len)[None, :]
    # Clamp labels to valid range for gathering (replace -100 with 0 temporarily)
    safe_labels = np.where(shift_labels == -100, 0, shift_labels)
    token_logits = shift_logits_f32[batch_idx, seq_idx, safe_labels]
    log_probs = token_logits - log_sum_exp

    masked_log_probs = np.where(mask, log_probs, 0.0)
    mean_neg_log_prob = -masked_log_probs.sum() / max(float(mask.sum()), 1.0)
    perplexity = float(np.exp(min(mean_neg_log_prob, 100.0)))  # cap to avoid inf

    return {"accuracy": accuracy, "perplexity": perplexity}


def get_training_args(
    output_dir: str, phase: int = 1, **overrides: Any
) -> TrainingArguments:
    """Build a ``TrainingArguments`` instance from config with phase-specific settings.

    Starts from the base ``TRAINING_ARGS`` dict defined in
    ``config/training_config.py``, applies phase-specific learning-rate and
    epoch overrides from ``PHASES``, and finally applies any caller-supplied
    keyword overrides.

    Args:
        output_dir: Directory for checkpoints and logs.
        phase: Training phase (1-4). Each phase carries its own learning rate
            and epoch count as defined in ``config.training_config.PHASES``.
        **overrides: Arbitrary keyword arguments forwarded to
            ``TrainingArguments``.  These take highest priority.

    Returns:
        A fully configured ``TrainingArguments`` instance.

    Raises:
        ValueError: If the phase number is not in the PHASES dict.
    """
    if phase not in PHASES:
        raise ValueError(
            f"Unknown training phase {phase}. Valid phases: {list(PHASES.keys())}"
        )

    phase_cfg = PHASES[phase]
    args = dict(TRAINING_ARGS)
    args["output_dir"] = output_dir
    args["learning_rate"] = phase_cfg["learning_rate"]
    args["num_train_epochs"] = phase_cfg["epochs"]
    args["run_name"] = f"phase{phase}_{phase_cfg['name'].lower()}"
    args["logging_dir"] = f"{output_dir}/logs"
    args["report_to"] = ["tensorboard"]

    # Apply caller overrides last so they always win
    args.update(overrides)

    logger.info(
        "Training args for phase %d (%s): lr=%s, epochs=%s, batch=%s, grad_accum=%s",
        phase,
        phase_cfg["name"],
        args["learning_rate"],
        args["num_train_epochs"],
        args["per_device_train_batch_size"],
        args["gradient_accumulation_steps"],
    )

    return TrainingArguments(**args)


def check_gpu_availability() -> dict[str, Any]:
    """Check CUDA availability and print GPU information.

    Returns:
        Dictionary containing:
        - ``available``: bool indicating CUDA availability.
        - ``device_count``: number of GPUs visible.
        - ``devices``: list of dicts with ``name``, ``vram_gb``, and
          ``compute_capability`` per GPU.
    """
    info: dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "device_count": 0,
        "devices": [],
    }

    if not info["available"]:
        logger.warning("CUDA is not available. Training will run on CPU (very slow).")
        print("WARNING: No CUDA GPUs detected. Training will be extremely slow on CPU.")
        return info

    info["device_count"] = torch.cuda.device_count()
    logger.info("CUDA available: %d GPU(s) detected", info["device_count"])
    print(f"CUDA available: {info['device_count']} GPU(s) detected")

    for i in range(info["device_count"]):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_mem / (1024**3)
        cc = f"{props.major}.{props.minor}"
        device_info = {
            "name": props.name,
            "vram_gb": round(vram_gb, 2),
            "compute_capability": cc,
        }
        info["devices"].append(device_info)
        print(
            f"  GPU {i}: {props.name} | "
            f"VRAM: {vram_gb:.1f} GB | "
            f"Compute Capability: {cc}"
        )

    # Current memory snapshot for first device
    if info["device_count"] > 0:
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print(
            f"  Memory (GPU 0): {allocated:.2f} GB allocated, {reserved:.2f} GB reserved"
        )

    return info


def estimate_training_time(
    num_samples: int,
    batch_size: int,
    num_epochs: int,
    samples_per_second: float = 40.0,
) -> dict[str, Any]:
    """Estimate wall-clock training time and approximate cloud compute cost.

    Args:
        num_samples: Total number of training samples.
        batch_size: Effective batch size (per-device * gradient accumulation *
            number of GPUs).
        num_epochs: Number of training epochs.
        samples_per_second: Estimated throughput.  Default of 40 is a
            conservative estimate for a 7B model with LoRA on an A40.

    Returns:
        Dictionary with ``total_steps``, ``estimated_seconds``,
        ``estimated_human`` (formatted string), and
        ``estimated_cost_usd`` (at $0.79/hr A40 rate).
    """
    steps_per_epoch = math.ceil(num_samples / batch_size)
    total_steps = steps_per_epoch * num_epochs
    total_samples = num_samples * num_epochs
    estimated_seconds = total_samples / samples_per_second

    hours = estimated_seconds / 3600
    # RunPod A40 pricing as rough estimate
    cost_per_hour = 0.79
    estimated_cost = hours * cost_per_hour

    # Human-readable duration
    if estimated_seconds < 60:
        human = f"{estimated_seconds:.0f} seconds"
    elif estimated_seconds < 3600:
        human = f"{estimated_seconds / 60:.1f} minutes"
    else:
        h = int(estimated_seconds // 3600)
        m = int((estimated_seconds % 3600) // 60)
        human = f"{h}h {m}m"

    result = {
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "estimated_seconds": round(estimated_seconds, 1),
        "estimated_human": human,
        "estimated_cost_usd": round(estimated_cost, 2),
    }

    print(
        f"Training estimate:\n"
        f"  Samples: {num_samples:,} x {num_epochs} epochs = {total_samples:,} total\n"
        f"  Steps: {total_steps:,} ({steps_per_epoch:,} per epoch)\n"
        f"  Time: ~{human} (at {samples_per_second:.0f} samples/sec)\n"
        f"  Cost: ~${estimated_cost:.2f} (A40 @ ${cost_per_hour}/hr)"
    )

    return result
