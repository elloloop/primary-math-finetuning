"""Custom training callbacks for GPU monitoring, early stopping, and progress display."""

import logging
import sys
import time
from typing import Optional

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class GPUMemoryCallback(TrainerCallback):
    """Log GPU memory usage at each logging step.

    Records allocated and reserved GPU memory to both the standard logger
    and the trainer's own log dict so values appear in TensorBoard.
    """

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        **kwargs,
    ):
        if not torch.cuda.is_available():
            return

        allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
        max_allocated_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        logger.info(
            "Step %d | GPU Memory: %.2f GB allocated, %.2f GB reserved, %.2f GB peak",
            state.global_step,
            allocated_gb,
            reserved_gb,
            max_allocated_gb,
        )

        if logs is not None:
            logs["gpu_memory_allocated_gb"] = round(allocated_gb, 3)
            logs["gpu_memory_reserved_gb"] = round(reserved_gb, 3)
            logs["gpu_memory_peak_gb"] = round(max_allocated_gb, 3)


class EarlyStoppingWithPatience(TrainerCallback):
    """Stop training when eval loss fails to improve for a given number of evaluations.

    Unlike the built-in ``EarlyStoppingCallback`` this version logs each
    decision and allows easy customisation of the metric and comparison
    direction.

    Args:
        patience: Number of evaluation rounds with no improvement before
            stopping.  Defaults to 3.
        min_delta: Minimum absolute improvement to count as progress.
            Defaults to 0.0.
        metric_key: The metric to monitor.  Defaults to ``"eval_loss"``.
    """

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0,
        metric_key: str = "eval_loss",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.metric_key = metric_key
        self.best_value: Optional[float] = None
        self.rounds_without_improvement = 0

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[dict] = None,
        **kwargs,
    ):
        if metrics is None:
            return

        current = metrics.get(self.metric_key)
        if current is None:
            logger.warning(
                "EarlyStopping: metric '%s' not found in evaluation metrics.",
                self.metric_key,
            )
            return

        if self.best_value is None or current < self.best_value - self.min_delta:
            improvement = (
                f"improved from {self.best_value:.6f}" if self.best_value is not None else "first evaluation"
            )
            logger.info(
                "EarlyStopping: %s = %.6f (%s). Resetting patience.",
                self.metric_key,
                current,
                improvement,
            )
            self.best_value = current
            self.rounds_without_improvement = 0
        else:
            self.rounds_without_improvement += 1
            logger.info(
                "EarlyStopping: %s = %.6f (no improvement). "
                "Patience: %d / %d.",
                self.metric_key,
                current,
                self.rounds_without_improvement,
                self.patience,
            )

        if self.rounds_without_improvement >= self.patience:
            logger.info(
                "EarlyStopping: patience exhausted (%d rounds). Stopping training.",
                self.patience,
            )
            control.should_training_stop = True


class TrainingProgressCallback(TrainerCallback):
    """Print a formatted progress bar at each logging step.

    Output example::

        Epoch 1/3 [===>  ] Step 100/1000 | Loss: 0.234 | LR: 1.98e-4 | 45.2 samples/sec

    The progress bar width is fixed at 20 characters for consistent formatting
    regardless of terminal width.
    """

    BAR_WIDTH = 20

    def __init__(self):
        self._train_start_time: Optional[float] = None
        self._last_log_time: Optional[float] = None
        self._last_log_step: int = 0

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self._train_start_time = time.time()
        self._last_log_time = time.time()
        self._last_log_step = 0
        print(f"\nTraining started | {state.max_steps} total steps | {args.num_train_epochs} epoch(s)")
        print("-" * 80)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        **kwargs,
    ):
        if logs is None or state.max_steps == 0:
            return

        step = state.global_step
        max_steps = state.max_steps
        epoch = state.epoch or 0.0
        num_epochs = args.num_train_epochs

        # Current epoch (1-indexed, clamped)
        current_epoch = min(int(epoch) + 1, int(num_epochs))

        # Progress bar
        fraction = step / max_steps
        filled = int(self.BAR_WIDTH * fraction)
        bar = "=" * filled
        if filled < self.BAR_WIDTH:
            bar += ">"
            bar += " " * (self.BAR_WIDTH - filled - 1)
        else:
            bar = "=" * self.BAR_WIDTH

        # Loss
        loss = logs.get("loss", logs.get("train_loss"))
        loss_str = f"{loss:.3f}" if loss is not None else "N/A"

        # Learning rate
        lr = logs.get("learning_rate")
        lr_str = f"{lr:.2e}" if lr is not None else "N/A"

        # Throughput: samples per second
        now = time.time()
        steps_delta = step - self._last_log_step
        time_delta = now - (self._last_log_time or now)
        if time_delta > 0 and steps_delta > 0:
            effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
            if torch.cuda.is_available():
                effective_batch *= max(torch.cuda.device_count(), 1)
            samples_per_sec = (steps_delta * effective_batch) / time_delta
        else:
            samples_per_sec = 0.0

        self._last_log_time = now
        self._last_log_step = step

        line = (
            f"Epoch {current_epoch}/{int(num_epochs)} "
            f"[{bar}] "
            f"Step {step}/{max_steps} | "
            f"Loss: {loss_str} | "
            f"LR: {lr_str} | "
            f"{samples_per_sec:.1f} samples/sec"
        )

        # Write to stderr so it doesn't interfere with piped output
        sys.stderr.write(f"\r{line}")
        sys.stderr.flush()

        # Also write a newline version to the log at coarser intervals
        if step % (args.logging_steps * 5) == 0 or step == max_steps:
            sys.stderr.write("\n")
            logger.info(line)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        elapsed = time.time() - (self._train_start_time or time.time())
        minutes = elapsed / 60
        sys.stderr.write("\n")
        print("-" * 80)
        print(
            f"Training complete | {state.global_step} steps | "
            f"{minutes:.1f} minutes | "
            f"Best metric: {state.best_metric}"
        )
