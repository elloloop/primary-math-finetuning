"""Data utilities package for primary math fine-tuning.

Provides dataset loading, synthetic data generation, ChatML formatting,
and data quality validation.
"""

from src.data.data_formatter import (
    DEFAULT_SYSTEM_MESSAGE,
    format_batch,
    format_for_training,
    to_chatml,
)
from src.data.data_generator import (
    augment_gsm8k,
    generate_distractors,
    generate_math_problems,
    generate_samples,
    save_samples,
)
from src.data.data_loader import (
    create_train_val_split,
    load_gsm8k,
    load_json_dataset,
    load_training_dataset,
)
from src.data.data_validator import (
    detect_duplicates,
    get_dataset_stats,
    validate_dataset,
    validate_record,
    validate_sample,
)

__all__ = [
    # data_loader
    "load_json_dataset",
    "load_gsm8k",
    "create_train_val_split",
    "load_training_dataset",
    # data_generator
    "generate_math_problems",
    "generate_distractors",
    "generate_samples",
    "augment_gsm8k",
    "save_samples",
    # data_formatter
    "format_for_training",
    "format_batch",
    "to_chatml",
    "DEFAULT_SYSTEM_MESSAGE",
    # data_validator
    "validate_sample",
    "validate_record",
    "validate_dataset",
    "detect_duplicates",
    "get_dataset_stats",
]
