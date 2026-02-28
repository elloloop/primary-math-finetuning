"""Main training orchestrator for primary-math fine-tuning."""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import Dataset, load_dataset
from peft import PeftModel
from trl import SFTTrainer

from config.model_config import LORA_CONFIG, MODEL_CONFIG
from config.training_config import PHASES
from src.data.data_formatter import format_for_training
from src.models.lora_config import apply_lora, get_lora_config, merge_and_save
from src.models.model_loader import get_tokenizer, load_base_model
from src.training.callbacks import (
    EarlyStoppingWithPatience,
    GPUMemoryCallback,
    TrainingProgressCallback,
)
from src.training.utils import (
    build_data_collator,
    check_gpu_availability,
    estimate_training_time,
    get_training_args,
)

logger = logging.getLogger(__name__)


class MathTrainer:
    """End-to-end orchestrator for LoRA fine-tuning of a causal LM on math data.

    Handles model loading, LoRA application, dataset preparation, training
    via TRL's ``SFTTrainer``, evaluation, and model saving (with optional
    LoRA merge).

    Args:
        model_name: HuggingFace model identifier or local path.
        output_dir: Root directory for checkpoints, logs, and saved models.
        phase: Training phase (1-4) that selects learning rate and epoch
            schedule from ``config.training_config.PHASES``.
        training_args_override: Optional dict of ``TrainingArguments`` fields
            to override beyond what the phase already sets.
        lora_config_override: Optional dict of LoRA hyperparameters to
            override the defaults in ``config.model_config.LORA_CONFIG``.
    """

    def __init__(
        self,
        model_name: str = MODEL_CONFIG["model_name"],
        output_dir: str = "outputs",
        phase: int = 1,
        training_args_override: Optional[dict[str, Any]] = None,
        lora_config_override: Optional[dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.phase = phase
        self.training_args_override = training_args_override or {}
        self.lora_config_override = lora_config_override or {}

        self.model = None
        self.tokenizer = None
        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None
        self.trainer: Optional[SFTTrainer] = None

        # Ensure output dir exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        phase_name = PHASES.get(phase, {}).get("name", "unknown")
        logger.info(
            "MathTrainer initialised: model=%s, phase=%d (%s), output=%s",
            model_name,
            phase,
            phase_name,
            output_dir,
        )

    # ------------------------------------------------------------------
    # Model preparation
    # ------------------------------------------------------------------

    def prepare_model(self) -> None:
        """Load the base model and tokenizer, then apply LoRA adapters.

        GPU availability is checked first.  If no CUDA device is found the
        method still proceeds (training will be slow on CPU).

        Raises:
            ValueError: If model or tokenizer loading fails.
            RuntimeError: Wraps CUDA OOM with actionable suggestions.
        """
        gpu_info = check_gpu_availability()

        logger.info("Loading tokenizer for %s", self.model_name)
        self.tokenizer = get_tokenizer(self.model_name)

        logger.info("Loading base model")
        try:
            self.model = load_base_model(
                model_name=self.model_name,
                device_map="auto" if gpu_info["available"] else "cpu",
                bf16=MODEL_CONFIG.get("bf16", True) and gpu_info["available"],
            )
        except torch.cuda.OutOfMemoryError as oom:
            self._handle_oom(oom)
            raise

        # Build LoRA config, merging defaults with any overrides
        lora_kwargs = dict(LORA_CONFIG)
        lora_kwargs.update(self.lora_config_override)
        # get_lora_config only accepts r, lora_alpha, lora_dropout
        lora_cfg = get_lora_config(
            r=lora_kwargs.get("r", 64),
            lora_alpha=lora_kwargs.get("lora_alpha", 128),
            lora_dropout=lora_kwargs.get("lora_dropout", 0.05),
        )

        logger.info("Applying LoRA adapters")
        self.model = apply_lora(self.model, lora_cfg)

        # Enable gradient checkpointing if configured
        if self.model is not None and hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

        logger.info("Model preparation complete")

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------

    def prepare_data(self, data_path: str, val_ratio: float = 0.1) -> None:
        """Load and format the dataset, splitting into train and validation sets.

        The data source can be:
        * A local JSON or JSONL file.
        * A HuggingFace dataset identifier.

        Each sample is formatted into Qwen ChatML via
        ``src.data.data_formatter.format_for_training`` and stored as a
        ``text`` column suitable for ``SFTTrainer``.

        Args:
            data_path: Path to a JSON/JSONL file or a HuggingFace dataset name.
            val_ratio: Fraction of data to reserve for validation (0.0-1.0).

        Raises:
            FileNotFoundError: If a local file path does not exist.
            ValueError: If the dataset is empty after formatting.
        """
        logger.info("Loading data from %s (val_ratio=%.2f)", data_path, val_ratio)

        # Load raw samples
        path = Path(data_path)
        if path.exists() and path.is_file():
            ext = path.suffix.lower()
            if ext == ".jsonl":
                ds = load_dataset("json", data_files=str(path), split="train")
            elif ext == ".json":
                ds = load_dataset("json", data_files=str(path), split="train")
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
        else:
            # Assume HuggingFace dataset identifier
            ds = load_dataset(data_path, split="train")

        logger.info("Loaded %d raw samples", len(ds))

        # Format each sample into ChatML text
        def _format(example):
            return format_for_training(example)

        ds = ds.map(_format, remove_columns=ds.column_names)

        if len(ds) == 0:
            raise ValueError("Dataset is empty after formatting")

        # Train/val split
        if val_ratio > 0:
            split = ds.train_test_split(test_size=val_ratio, seed=42)
            self.train_dataset = split["train"]
            self.eval_dataset = split["test"]
        else:
            self.train_dataset = ds
            self.eval_dataset = None

        train_count = len(self.train_dataset)
        eval_count = len(self.eval_dataset) if self.eval_dataset else 0
        logger.info("Data prepared: %d train, %d eval samples", train_count, eval_count)

        # Print training time estimate
        ta = get_training_args(
            self.output_dir, phase=self.phase, **self.training_args_override
        )
        effective_batch = (
            ta.per_device_train_batch_size
            * ta.gradient_accumulation_steps
            * max(torch.cuda.device_count(), 1)
            if torch.cuda.is_available()
            else ta.per_device_train_batch_size * ta.gradient_accumulation_steps
        )
        estimate_training_time(
            num_samples=train_count,
            batch_size=effective_batch,
            num_epochs=int(ta.num_train_epochs),
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, resume_from_checkpoint: Optional[str] = None) -> dict[str, Any]:
        """Run training using TRL's SFTTrainer.

        Args:
            resume_from_checkpoint: Optional path to a checkpoint directory
                to resume training from.

        Returns:
            The ``TrainOutput`` metrics dictionary.

        Raises:
            RuntimeError: If model or data has not been prepared.
            RuntimeError: Wraps CUDA OOM with actionable suggestions.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Model not prepared. Call prepare_model() before train()."
            )
        if self.train_dataset is None:
            raise RuntimeError(
                "Training data not prepared. Call prepare_data() before train()."
            )

        training_args = get_training_args(
            output_dir=self.output_dir,
            phase=self.phase,
            **self.training_args_override,
        )

        callbacks = [
            GPUMemoryCallback(),
            TrainingProgressCallback(),
        ]
        if self.eval_dataset is not None:
            callbacks.append(EarlyStoppingWithPatience(patience=3, min_delta=0.001))

        logger.info(
            "Initialising SFTTrainer for phase %d (%s)",
            self.phase,
            PHASES[self.phase]["name"],
        )

        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
            data_collator=build_data_collator(self.tokenizer),
            callbacks=callbacks,
            max_seq_length=MODEL_CONFIG.get("max_seq_length", 2048),
        )

        logger.info(
            "Starting training (resume_from_checkpoint=%s)", resume_from_checkpoint
        )
        try:
            result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        except torch.cuda.OutOfMemoryError as oom:
            self._handle_oom(oom)
            raise

        metrics = result.metrics
        logger.info("Training complete. Metrics: %s", metrics)

        # Save training metrics
        metrics_path = Path(self.output_dir) / "train_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Training metrics saved to %s", metrics_path)

        return metrics

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> dict[str, float]:
        """Run evaluation on the validation set.

        Returns:
            Dictionary of evaluation metrics.

        Raises:
            RuntimeError: If trainer or eval data is not available.
        """
        if self.trainer is None:
            raise RuntimeError(
                "Trainer not initialised. Call train() before evaluate()."
            )
        if self.eval_dataset is None:
            raise RuntimeError("No evaluation dataset available.")

        logger.info("Running evaluation on %d samples", len(self.eval_dataset))
        metrics = self.trainer.evaluate()
        logger.info("Evaluation metrics: %s", metrics)

        # Save eval metrics
        metrics_path = Path(self.output_dir) / "eval_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Evaluation metrics saved to %s", metrics_path)

        return metrics

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, merge_lora: bool = True) -> str:
        """Save the trained model and tokenizer.

        Args:
            merge_lora: If True and the model is a PeftModel, merge LoRA
                weights into the base model before saving.  If False, save
                the adapter weights only (smaller, but requires PEFT to load).

        Returns:
            Path to the saved model directory.

        Raises:
            RuntimeError: If model or tokenizer is not available.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Model not available. Call prepare_model() and train() first."
            )

        save_dir = str(Path(self.output_dir) / "final")

        if merge_lora and isinstance(self.model, PeftModel):
            logger.info("Merging LoRA weights and saving to %s", save_dir)
            merge_and_save(self.model, self.tokenizer, save_dir)
        else:
            logger.info("Saving model to %s (merge_lora=%s)", save_dir, merge_lora)
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)

        logger.info("Model saved to %s", save_dir)
        return save_dir

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _handle_oom(error: torch.cuda.OutOfMemoryError) -> None:
        """Log actionable advice when a CUDA OOM error occurs."""
        logger.error("CUDA Out of Memory: %s", error)
        print(
            "\n"
            "========================================\n"
            "  CUDA OUT OF MEMORY\n"
            "========================================\n"
            "Suggestions to reduce memory usage:\n"
            "  1. Reduce per_device_train_batch_size (current default: 8)\n"
            "  2. Increase gradient_accumulation_steps to compensate\n"
            "  3. Reduce max_seq_length (current default: 2048)\n"
            "  4. Enable gradient_checkpointing (should be on by default)\n"
            "  5. Use a smaller LoRA rank (r): try 32 instead of 64\n"
            "  6. Ensure no other processes are using GPU memory\n"
            "  7. Try a smaller base model\n"
            "========================================\n",
            file=sys.stderr,
        )
        # Free cached memory to allow graceful shutdown
        torch.cuda.empty_cache()
