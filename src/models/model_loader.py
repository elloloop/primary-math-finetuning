"""Model loading utilities for base and fine-tuned models."""

import logging
import torch
from peft import AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


def _report_gpu_memory(label: str) -> None:
    """Report current GPU memory usage.

    Args:
        label: Label to identify the memory report in logs.
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(
            f"{label} - GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )


def get_tokenizer(model_name: str = "Qwen/Qwen2.5-7B-Instruct") -> PreTrainedTokenizer:
    """Load tokenizer with proper configuration.

    Loads the tokenizer for the specified model and ensures proper pad token setup
    for training.

    Args:
        model_name: Hugging Face model identifier.

    Returns:
        Configured PreTrainedTokenizer instance.

    Raises:
        ValueError: If tokenizer cannot be loaded.
    """
    try:
        logger.info(f"Loading tokenizer from {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")

        logger.info(f"Tokenizer loaded successfully. Vocab size: {len(tokenizer)}")
        return tokenizer

    except Exception as e:
        error_msg = f"Failed to load tokenizer from {model_name}: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def load_base_model(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    device_map: str = "auto",
    bf16: bool = True,
) -> PreTrainedModel:
    """Load base model with appropriate settings for A40 GPU.

    Loads the base model with automatic device placement and optional bfloat16
    precision for memory efficiency on GPUs like A40.

    Args:
        model_name: Hugging Face model identifier.
        device_map: Device mapping strategy ("auto", "cpu", "cuda", etc.).
        bf16: Use bfloat16 precision for reduced memory usage.

    Returns:
        Loaded PreTrainedModel instance.

    Raises:
        ValueError: If model cannot be loaded.
    """
    try:
        logger.info(f"Loading base model {model_name}")
        logger.info(f"Configuration: device_map={device_map}, bf16={bf16}")

        _report_gpu_memory("Before model loading")

        torch_dtype = torch.bfloat16 if bf16 else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
            if torch.cuda.is_available()
            else None,
        )

        _report_gpu_memory("After model loading")
        logger.info(f"Model loaded successfully: {model.config.model_type}")

        return model

    except Exception as e:
        error_msg = f"Failed to load base model {model_name}: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def load_finetuned_model(
    model_path: str,
    device_map: str = "auto",
) -> PreTrainedModel:
    """Load a fine-tuned model (merged or with LoRA adapter).

    Attempts to load a model that may be either:
    1. A merged model (base model with LoRA weights integrated)
    2. A model with LoRA adapter weights to be loaded on top of base model

    Args:
        model_path: Path to the fine-tuned model directory.
        device_map: Device mapping strategy.

    Returns:
        Loaded PreTrainedModel instance (with LoRA adapters if applicable).

    Raises:
        ValueError: If model cannot be loaded.
    """
    try:
        logger.info(f"Loading fine-tuned model from {model_path}")

        _report_gpu_memory("Before fine-tuned model loading")

        try:
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            logger.info("Loaded model with LoRA adapters")
        except Exception:
            logger.info("Could not load as PEFT model, attempting standard load")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            logger.info("Loaded as merged/base model")

        _report_gpu_memory("After fine-tuned model loading")
        logger.info("Fine-tuned model loaded successfully")

        return model

    except Exception as e:
        error_msg = f"Failed to load fine-tuned model from {model_path}: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e
