"""Base evaluator for language model benchmarks."""

import re
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class BaseEvaluator(ABC):
    """Abstract base evaluator providing generation and answer extraction utilities.

    Subclasses must implement the ``evaluate`` method for benchmark-specific
    evaluation logic.

    Args:
        model: A HuggingFace pretrained language model.
        tokenizer: The tokenizer corresponding to ``model``.
        max_new_tokens: Maximum number of tokens to generate per prompt.
        temperature: Sampling temperature. Use 0.0 for greedy decoding.
        batch_size: Number of prompts to process in a single forward pass.
    """

    # Ordered list of regex patterns used to extract a final answer from
    # generated text. Earlier patterns take priority.
    ANSWER_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"####\s*(-?\d[\d,]*\.?\d*)", re.IGNORECASE),
        re.compile(r"(?:the\s+)?answer\s+is\s*:?\s*([A-Da-d])\b", re.IGNORECASE),
        re.compile(r"correct\s+answer\s+is\s*:?\s*([A-Da-d])\b", re.IGNORECASE),
        re.compile(r"^\s*([A-Da-d])\)", re.MULTILINE),
        re.compile(r"(-?\d[\d,]*\.?\d*)\s*$", re.MULTILINE),
    ]

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        batch_size: int = 8,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.batch_size = batch_size

        # Ensure the tokenizer has a pad token for batched generation.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.device = next(model.parameters()).device

    def generate(self, prompts: list[str]) -> list[str]:
        """Generate responses for a batch of prompts.

        Prompts are processed in chunks of ``self.batch_size``. Greedy
        decoding is used when ``temperature == 0.0``; otherwise nucleus
        sampling is applied.

        Args:
            prompts: A list of prompt strings.

        Returns:
            A list of generated response strings, one per input prompt.
        """
        all_responses: list[str] = []

        for start in range(0, len(prompts), self.batch_size):
            batch = prompts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.device)

            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": self.max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
            }

            if self.temperature == 0.0:
                generation_kwargs["do_sample"] = False
            else:
                generation_kwargs["do_sample"] = True
                generation_kwargs["temperature"] = self.temperature
                generation_kwargs["top_p"] = 0.95

            with torch.no_grad():
                output_ids = self.model.generate(**encoded, **generation_kwargs)

            # Decode only the newly generated tokens (strip the input portion).
            for i, output in enumerate(output_ids):
                input_length = encoded["input_ids"][i].shape[0]
                generated_tokens = output[input_length:]
                response = self.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                )
                all_responses.append(response.strip())

        return all_responses

    def extract_answer(self, response: str) -> Optional[str]:
        """Extract the final answer from a model-generated response.

        Tries each pattern in ``ANSWER_PATTERNS`` in order and returns the
        first match. Numeric answers have commas stripped; letter answers are
        upper-cased.

        Args:
            response: The raw text output from the model.

        Returns:
            The extracted answer string, or ``None`` if no pattern matched.
        """
        for pattern in self.ANSWER_PATTERNS:
            match = pattern.search(response)
            if match:
                raw = match.group(1).strip()
                # Remove commas from numeric answers.
                raw = raw.replace(",", "")
                # Upper-case single-letter answers for consistency.
                if len(raw) == 1 and raw.isalpha():
                    return raw.upper()
                return raw
        return None

    def _normalize_answer(self, answer: str) -> str:
        """Normalize an answer string for comparison.

        Applies the following transformations:
        - Strip leading/trailing whitespace.
        - Convert to lowercase.
        - Remove commas from numbers.
        - Convert float-like integers (e.g. ``"42.0"``) to plain integers
          (``"42"``).

        Args:
            answer: The raw answer string.

        Returns:
            The normalized answer string.
        """
        answer = answer.strip().lower()
        answer = answer.replace(",", "")

        # Convert "42.0" -> "42", "3.00" -> "3", but leave "3.14" as-is.
        try:
            numeric = float(answer)
            if numeric == int(numeric):
                answer = str(int(numeric))
            else:
                answer = str(numeric)
        except ValueError:
            pass

        return answer

    @abstractmethod
    def evaluate(self, dataset: Any) -> dict[str, Any]:
        """Run evaluation on a dataset and return a results dictionary.

        Subclasses must implement this method.

        Args:
            dataset: The evaluation dataset (format depends on subclass).

        Returns:
            A dictionary containing evaluation metrics and per-example
            predictions.
        """
        ...
