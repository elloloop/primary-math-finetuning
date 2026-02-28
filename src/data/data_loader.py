"""Dataset loading utilities for primary math fine-tuning.

Supports loading from local JSON/JSONL files and HuggingFace datasets,
with train/validation splitting functionality.
"""

import json
import logging
from pathlib import Path
from typing import Any, Union

from datasets import Dataset, DatasetDict, load_dataset

from src.data.data_formatter import to_chatml
from src.data.data_validator import validate_dataset as _validate_dataset

logger = logging.getLogger(__name__)


def load_json_dataset(path: Union[str, Path]) -> list[dict[str, Any]]:
    """Load training data from a JSON or JSONL file.

    Supports two formats:
    - JSON: a single JSON array of records.
    - JSONL: one JSON object per line.

    Args:
        path: Path to the JSON or JSONL file.

    Returns:
        A list of sample dictionaries.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported or contents are invalid.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    suffix = filepath.suffix.lower()
    if suffix not in {".json", ".jsonl"}:
        raise ValueError(
            f"Unsupported file format '{suffix}'. Expected .json or .jsonl"
        )

    records: list[dict[str, Any]] = []

    if suffix == ".jsonl":
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_num} of {filepath}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Line {line_num} of {filepath} is not a JSON object"
                    )
                records.append(record)
    else:
        raw = filepath.read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {filepath}: {exc}") from exc

        if isinstance(data, list):
            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    raise ValueError(
                        f"Item at index {i} in {filepath} is not a JSON object"
                    )
            records = data
        elif isinstance(data, dict):
            records = [data]
        else:
            raise ValueError(
                f"Expected a JSON array or object in {filepath}, got {type(data).__name__}"
            )

    logger.info("Loaded %d records from %s", len(records), filepath)
    return records


def load_gsm8k(split: str = "train") -> Dataset:
    """Load the GSM8K dataset from HuggingFace.

    GSM8K (Grade School Math 8K) is a dataset of high-quality grade school
    math word problems. Each problem requires 2-8 steps to solve and the
    final answer is a single numeric value.

    Args:
        split: Which split to load. One of 'train', 'test', or 'all'.
            Use 'all' to get a DatasetDict with both splits.

    Returns:
        A HuggingFace Dataset containing the requested split.

    Raises:
        ValueError: If an invalid split name is provided.
    """
    valid_splits = {"train", "test"}
    if split not in valid_splits:
        raise ValueError(
            f"Invalid split '{split}'. Must be one of {sorted(valid_splits)}"
        )

    logger.info("Loading GSM8K dataset (split=%s) from HuggingFace...", split)
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    logger.info("Loaded %d samples from GSM8K %s split", len(dataset), split)
    return dataset


def create_train_val_split(
    dataset: Union[Dataset, list[dict[str, Any]]],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """Split a dataset into training and validation sets.

    Args:
        dataset: A HuggingFace Dataset or a list of dictionaries.
        val_ratio: Fraction of data to use for validation. Must be in (0, 1).
        seed: Random seed for reproducible splits.

    Returns:
        A DatasetDict with 'train' and 'validation' keys.

    Raises:
        ValueError: If val_ratio is out of range or dataset is empty.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(
            f"val_ratio must be between 0 and 1 exclusive, got {val_ratio}"
        )

    if isinstance(dataset, list):
        if len(dataset) == 0:
            raise ValueError("Cannot split an empty dataset")
        dataset = Dataset.from_list(dataset)

    if len(dataset) == 0:
        raise ValueError("Cannot split an empty dataset")

    min_samples = max(2, int(1.0 / val_ratio) + 1)
    if len(dataset) < min_samples:
        raise ValueError(
            f"Dataset has {len(dataset)} samples, need at least {min_samples} "
            f"for a {val_ratio:.0%} validation split"
        )

    split_dataset = dataset.train_test_split(test_size=val_ratio, seed=seed)

    result = DatasetDict(
        {
            "train": split_dataset["train"],
            "validation": split_dataset["test"],
        }
    )

    logger.info(
        "Split dataset: %d train, %d validation (val_ratio=%.2f, seed=%d)",
        len(result["train"]),
        len(result["validation"]),
        val_ratio,
        seed,
    )
    return result


def load_training_dataset(path: str, quick_test: bool = False) -> Dataset:
    """Load and validate a JSON training dataset, returning ChatML-formatted data.

    This is a convenience function that loads a JSON file, validates its
    contents, formats each record to ChatML, and returns a HuggingFace
    Dataset ready for SFTTrainer.

    Args:
        path: Path to a JSON file containing a list of sample dicts.
        quick_test: If True, only load the first 100 records.

    Returns:
        A HuggingFace Dataset with a 'text' column containing ChatML strings.

    Raises:
        ValueError: If dataset validation fails.
    """
    ok, errs = _validate_dataset(path)
    if not ok:
        raise ValueError("Dataset validation failed:\n" + "\n".join(errs[:20]))

    records = load_json_dataset(path)
    if quick_test:
        records = records[:100]

    texts = [{"text": to_chatml(r)} for r in records]
    return Dataset.from_list(texts)
