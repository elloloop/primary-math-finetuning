"""Data quality validation for math problem datasets.

Provides schema validation, duplicate detection, and statistical
analysis for training data quality assurance.
"""

import json
import logging
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {"question", "choices", "answer"}
VALID_ANSWERS = {"A", "B", "C", "D"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def validate_sample(sample: dict[str, Any]) -> list[str]:
    """Validate a single sample against the expected schema.

    Checks that the sample:
    - Contains all required fields (question, choices, answer).
    - Has exactly 4 unique choices.
    - Has an answer value in {A, B, C, D}.
    - Has a non-empty question string.
    - Has optional fields with correct types when present.

    Args:
        sample: A sample dictionary to validate.

    Returns:
        A list of error messages. An empty list indicates the sample
        is valid.
    """
    errors: list[str] = []

    # Check required fields
    missing = REQUIRED_FIELDS - set(sample.keys())
    if missing:
        errors.append(f"Missing required fields: {sorted(missing)}")
        return errors  # Cannot validate further without required fields

    # Validate question
    question = sample["question"]
    if not isinstance(question, str) or not question.strip():
        errors.append("'question' must be a non-empty string")

    # Validate choices
    choices = sample["choices"]
    if not isinstance(choices, list):
        errors.append(f"'choices' must be a list, got {type(choices).__name__}")
    elif len(choices) != 4:
        errors.append(f"'choices' must contain exactly 4 items, got {len(choices)}")
    else:
        for i, choice in enumerate(choices):
            if not isinstance(choice, str):
                errors.append(
                    f"'choices[{i}]' must be a string, got {type(choice).__name__}"
                )
        if len(set(choices)) != 4:
            errors.append("'choices' must contain 4 unique options")

    # Validate answer
    answer = sample["answer"]
    if answer not in VALID_ANSWERS:
        errors.append(f"'answer' must be one of {sorted(VALID_ANSWERS)}, got '{answer}'")

    # Validate optional fields
    if "explanation" in sample:
        if not isinstance(sample["explanation"], str):
            errors.append("'explanation' must be a string")

    if "difficulty" in sample:
        if sample["difficulty"] not in VALID_DIFFICULTIES:
            errors.append(
                f"'difficulty' must be one of {sorted(VALID_DIFFICULTIES)}, "
                f"got '{sample['difficulty']}'"
            )

    if "category" in sample:
        if not isinstance(sample["category"], str) or not sample["category"].strip():
            errors.append("'category' must be a non-empty string")

    return errors


def validate_dataset(
    samples: Union[list[dict[str, Any]], str, Path],
) -> tuple[bool, list[str]]:
    """Validate an entire dataset and return a detailed error report.

    Accepts either a list of sample dicts or a path to a JSON file.

    Args:
        samples: A list of sample dictionaries, or a path to a JSON file
            containing a list of samples.

    Returns:
        A tuple of (is_valid, errors) where is_valid is True if no errors
        were found, and errors is a list of error strings with record
        indices.
    """
    if isinstance(samples, (str, Path)):
        path = Path(samples)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return False, [f"Expected a JSON array in {path}, got {type(data).__name__}"]
        samples = data

    all_errors: list[str] = []
    error_counts: Counter = Counter()
    seen_questions: set[str] = set()

    for i, sample in enumerate(samples):
        if not isinstance(sample, dict):
            all_errors.append(f"Record {i}: expected a dictionary, got {type(sample).__name__}")
            error_counts["invalid_type"] += 1
            continue

        record_errors = validate_sample(sample)

        # Check for exact duplicates
        question = sample.get("question", "")
        if question in seen_questions:
            record_errors.append("Duplicate question")
            error_counts["duplicate"] += 1
        seen_questions.add(question)

        for err in record_errors:
            all_errors.append(f"Record {i}: {err}")
            # Categorize the error
            if "Missing" in err:
                error_counts["missing_fields"] += 1
            elif "choices" in err:
                error_counts["invalid_choices"] += 1
            elif "answer" in err:
                error_counts["invalid_answer"] += 1
            else:
                error_counts["other"] += 1

    is_valid = len(all_errors) == 0

    if is_valid:
        logger.info("Dataset validation passed: %d samples, 0 errors", len(samples))
    else:
        logger.warning(
            "Dataset validation failed: %d samples, %d errors. "
            "Breakdown: %s",
            len(samples),
            len(all_errors),
            dict(error_counts),
        )

    return is_valid, all_errors


def detect_duplicates(
    samples: list[dict[str, Any]],
    similarity_threshold: float = 0.85,
) -> list[tuple[int, int, float]]:
    """Find near-duplicate questions in the dataset.

    Uses SequenceMatcher to compute similarity ratios between question
    texts. Pairs exceeding the threshold are flagged.

    Args:
        samples: A list of sample dictionaries.
        similarity_threshold: Minimum similarity ratio (0-1) to flag a pair
            as near-duplicate. Defaults to 0.85.

    Returns:
        A list of (index_a, index_b, similarity_score) tuples for each
        pair of near-duplicate questions, sorted by descending similarity.
    """
    questions = [s.get("question", "") for s in samples]
    duplicates: list[tuple[int, int, float]] = []

    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            # Quick length-based pre-filter
            len_i, len_j = len(questions[i]), len(questions[j])
            if len_i == 0 or len_j == 0:
                continue
            if min(len_i, len_j) / max(len_i, len_j) < similarity_threshold:
                continue

            ratio = SequenceMatcher(None, questions[i], questions[j]).ratio()
            if ratio >= similarity_threshold:
                duplicates.append((i, j, round(ratio, 4)))

    duplicates.sort(key=lambda x: x[2], reverse=True)

    if duplicates:
        logger.info(
            "Found %d near-duplicate pairs (threshold=%.2f)",
            len(duplicates),
            similarity_threshold,
        )
    else:
        logger.info(
            "No near-duplicates found (threshold=%.2f)", similarity_threshold
        )

    return duplicates


def get_dataset_stats(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute descriptive statistics for a dataset.

    Args:
        samples: A list of sample dictionaries.

    Returns:
        A dictionary containing:
        - total_samples: Total number of samples.
        - by_difficulty: Count per difficulty level.
        - by_category: Count per category.
        - by_answer: Count per answer letter.
        - avg_question_length: Average character length of questions.
        - avg_choices_length: Average character length across all choices.
        - has_explanation: Count of samples with non-empty explanations.
        - missing_fields: Count of samples missing optional fields.
    """
    total = len(samples)
    if total == 0:
        return {
            "total_samples": 0,
            "by_difficulty": {},
            "by_category": {},
            "by_answer": {},
            "avg_question_length": 0.0,
            "avg_choices_length": 0.0,
            "has_explanation": 0,
            "missing_fields": {},
        }

    difficulty_counts: Counter = Counter()
    category_counts: Counter = Counter()
    answer_counts: Counter = Counter()
    question_lengths: list[int] = []
    choice_lengths: list[int] = []
    has_explanation_count = 0
    missing_optional: Counter = Counter()

    optional_fields = {"explanation", "difficulty", "category"}

    for sample in samples:
        difficulty_counts[sample.get("difficulty", "unknown")] += 1
        category_counts[sample.get("category", "unknown")] += 1
        answer_counts[sample.get("answer", "unknown")] += 1

        question = sample.get("question", "")
        question_lengths.append(len(question))

        choices = sample.get("choices", [])
        for choice in choices:
            choice_lengths.append(len(str(choice)))

        explanation = sample.get("explanation", "")
        if explanation and explanation.strip():
            has_explanation_count += 1

        for field in optional_fields:
            if field not in sample:
                missing_optional[field] += 1

    stats: dict[str, Any] = {
        "total_samples": total,
        "by_difficulty": dict(difficulty_counts.most_common()),
        "by_category": dict(category_counts.most_common()),
        "by_answer": dict(answer_counts.most_common()),
        "avg_question_length": round(sum(question_lengths) / total, 1),
        "avg_choices_length": round(
            sum(choice_lengths) / max(len(choice_lengths), 1), 1
        ),
        "has_explanation": has_explanation_count,
        "missing_fields": dict(missing_optional) if missing_optional else {},
    }

    logger.info(
        "Dataset stats: %d samples, %d difficulties, %d categories",
        total,
        len(difficulty_counts),
        len(category_counts),
    )
    return stats


# Backward-compatible alias
validate_record = validate_sample
