"""Format conversion for Qwen ChatML training data.

Converts structured math problem samples into the ChatML format expected
by Qwen models and compatible with the TRL SFTTrainer.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_MESSAGE = (
    "You are a helpful math tutor. Solve problems step by step "
    "and provide the correct answer."
)

_CHATML_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{question}\n\n"
    "Options:\n"
    "A) {c1}\n"
    "B) {c2}\n"
    "C) {c3}\n"
    "D) {c4}<|im_end|>\n"
    "<|im_start|>assistant\n{explanation}\n"
    "The correct answer is: {answer}<|im_end|>"
)


def format_for_training(
    sample: dict[str, Any],
    system_message: Optional[str] = None,
) -> dict[str, str]:
    """Convert a single sample into Qwen ChatML format for SFTTrainer.

    The output is a dictionary with a single 'text' key containing the
    fully formatted ChatML string, suitable for direct use with
    trl.SFTTrainer.

    Args:
        sample: A sample dictionary with keys 'question', 'choices' (list
            of 4 strings), 'answer' (one of 'A','B','C','D'), and
            optionally 'explanation'.
        system_message: The system prompt. Defaults to a standard math
            tutor message if not provided.

    Returns:
        A dictionary with key 'text' containing the ChatML-formatted string.

    Raises:
        KeyError: If required fields are missing from the sample.
        ValueError: If choices does not contain exactly 4 items.
    """
    if system_message is None:
        system_message = DEFAULT_SYSTEM_MESSAGE

    question = sample["question"]
    choices = sample["choices"]
    answer = sample["answer"]
    explanation = sample.get("explanation", "")

    if len(choices) != 4:
        raise ValueError(f"Expected exactly 4 choices, got {len(choices)}: {choices}")

    text = _CHATML_TEMPLATE.format(
        system=system_message,
        question=question,
        c1=choices[0],
        c2=choices[1],
        c3=choices[2],
        c4=choices[3],
        explanation=explanation,
        answer=answer,
    )

    return {"text": text}


def format_batch(
    samples: list[dict[str, Any]],
    system_message: Optional[str] = None,
) -> list[dict[str, str]]:
    """Convert a batch of samples into Qwen ChatML format.

    Args:
        samples: A list of sample dictionaries.
        system_message: The system prompt applied to every sample.
            Defaults to a standard math tutor message if not provided.

    Returns:
        A list of dictionaries, each with a 'text' key containing
        the ChatML-formatted string.

    Raises:
        KeyError: If required fields are missing from any sample.
        ValueError: If any sample's choices do not contain exactly 4 items.
    """
    formatted: list[dict[str, str]] = []
    for i, sample in enumerate(samples):
        try:
            formatted.append(format_for_training(sample, system_message))
        except (KeyError, ValueError) as exc:
            raise type(exc)(f"Error formatting sample at index {i}: {exc}") from exc

    logger.info("Formatted %d samples to ChatML", len(formatted))
    return formatted


def to_chatml(record: dict[str, Any], system_message: Optional[str] = None) -> str:
    """Convert a record to a ChatML string.

    This is a convenience wrapper around format_for_training that returns
    the raw text string instead of a dictionary. Maintained for backward
    compatibility.

    Args:
        record: A sample dictionary.
        system_message: Optional system prompt override.

    Returns:
        The ChatML-formatted string.
    """
    return format_for_training(record, system_message)["text"]
