"""LoRA configuration and utilities for model fine-tuning."""

import logging
from pathlib import Path
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


def get_lora_config(
    r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
) -> LoraConfig:
    """Create LoRA configuration for Qwen2.5 model.

    Configures LoRA adapters with specified hyperparameters for all attention
    and feedforward projections in the Qwen2.5 architecture.

    Args:
        r: LoRA rank (dimension of weight update matrices).
        lora_alpha: LoRA scaling factor (effective learning rate multiplier).
        lora_dropout: Dropout probability for LoRA layers.

    Returns:
        Configured LoraConfig object.
    """
    logger.info(
        f"Creating LoRA config: r={r}, lora_alpha={lora_alpha}, lora_dropout={lora_dropout}"
    )

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    logger.info(f"LoRA config created with target modules: {target_modules}")
    return config


def apply_lora(model: PreTrainedModel, config: LoraConfig) -> PeftModel:
    """Apply LoRA adapters to model.

    Wraps the model with PEFT LoRA configuration and reports trainable
    parameter statistics.

    Args:
        model: The base PreTrainedModel to apply LoRA to.
        config: LoraConfig object specifying LoRA parameters.

    Returns:
        Model wrapped with LoRA adapters (PeftModel).

    Raises:
        ValueError: If model is already a PEFT model or adapter application fails.
    """
    try:
        if isinstance(model, PeftModel):
            raise ValueError("Model is already a PEFT model")

        logger.info("Applying LoRA adapters to model")
        peft_model = get_peft_model(model, config)

        total_params = sum(p.numel() for p in peft_model.parameters())
        trainable_params = sum(
            p.numel() for p in peft_model.parameters() if p.requires_grad
        )
        trainable_percent = 100 * trainable_params / total_params

        logger.info(
            f"LoRA applied successfully. Trainable parameters: "
            f"{trainable_params:,} / {total_params:,} ({trainable_percent:.2f}%)"
        )

        return peft_model

    except Exception as e:
        error_msg = f"Failed to apply LoRA adapters: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def merge_and_save(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
) -> None:
    """Merge LoRA weights into base model and save.

    Merges LoRA adapter weights into the base model weights, integrating
    them permanently, and saves the merged model and tokenizer to disk.

    Args:
        model: PEFT model with LoRA adapters to merge.
        tokenizer: Tokenizer to save alongside model.
        output_dir: Directory path to save merged model.

    Raises:
        ValueError: If model is not a PEFT model or saving fails.
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if not isinstance(model, PeftModel):
            raise ValueError("Model must be a PEFT model to merge LoRA weights")

        logger.info("Merging LoRA weights into base model")
        merged_model = model.merge_and_unload()

        logger.info(f"Saving merged model to {output_dir}")
        merged_model.save_pretrained(str(output_path))

        logger.info("Saving tokenizer")
        tokenizer.save_pretrained(str(output_path))

        logger.info(f"Model and tokenizer saved successfully to {output_dir}")

    except Exception as e:
        error_msg = f"Failed to merge and save model: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e
