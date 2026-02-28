"""Synthetic math data generation for primary-level word problems.

Generates realistic multiple-choice word problems covering addition,
subtraction, multiplication, division, and mixed operations across
easy, medium, and hard difficulty levels.
"""

import json
import logging
import random
from typing import Any, Optional

from datasets import Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Problem templates organized by category
# ---------------------------------------------------------------------------

_ADDITION_TEMPLATES = [
    (
        "{name} had {a} {item}. {name2} gave {name} {b} more {item}. "
        "How many {item} does {name} have now?",
        lambda a, b: a + b,
        "{a} + {b} = {answer}",
    ),
    (
        "There are {a} {item} in one box and {b} {item} in another box. "
        "How many {item} are there in total?",
        lambda a, b: a + b,
        "{a} + {b} = {answer}",
    ),
    (
        "{name} read {a} pages on Monday and {b} pages on Tuesday. "
        "How many pages did {name} read in total?",
        lambda a, b: a + b,
        "{a} + {b} = {answer}",
    ),
    (
        "A store sold {a} {item} in the morning and {b} {item} in the afternoon. "
        "How many {item} were sold that day?",
        lambda a, b: a + b,
        "{a} + {b} = {answer}",
    ),
]

_SUBTRACTION_TEMPLATES = [
    (
        "{name} had {a} {item}. {name} gave {b} {item} to {name2}. "
        "How many {item} does {name} have left?",
        lambda a, b: a - b,
        "{a} - {b} = {answer}",
    ),
    (
        "There were {a} {item} on the shelf. {b} {item} were taken away. "
        "How many {item} are left on the shelf?",
        lambda a, b: a - b,
        "{a} - {b} = {answer}",
    ),
    (
        "{name} had {a} stickers. {name} used {b} stickers for a project. "
        "How many stickers does {name} have remaining?",
        lambda a, b: a - b,
        "{a} - {b} = {answer}",
    ),
    (
        "A bus had {a} passengers. At the next stop, {b} passengers got off. "
        "How many passengers are still on the bus?",
        lambda a, b: a - b,
        "{a} - {b} = {answer}",
    ),
]

_MULTIPLICATION_TEMPLATES = [
    (
        "{name} has {a} bags with {b} {item} in each bag. "
        "How many {item} does {name} have in total?",
        lambda a, b: a * b,
        "{a} x {b} = {answer}",
    ),
    (
        "There are {a} rows of {item} with {b} {item} in each row. "
        "How many {item} are there in total?",
        lambda a, b: a * b,
        "{a} x {b} = {answer}",
    ),
    (
        "Each box contains {b} {item}. If {name} buys {a} boxes, "
        "how many {item} will {name} have?",
        lambda a, b: a * b,
        "{a} x {b} = {answer}",
    ),
    (
        "{name} earns ${b} per hour. If {name} works for {a} hours, "
        "how much will {name} earn?",
        lambda a, b: a * b,
        "{a} x {b} = {answer}",
    ),
]

_DIVISION_TEMPLATES = [
    (
        "{name} has {a} {item} and wants to share them equally among "
        "{b} friends. How many {item} does each friend get?",
        lambda a, b: a // b,
        "{a} / {b} = {answer}",
    ),
    (
        "There are {a} {item} to be packed into boxes of {b}. "
        "How many full boxes can be made?",
        lambda a, b: a // b,
        "{a} / {b} = {answer}",
    ),
    (
        "{name} drove {a} miles over {b} days, traveling the same distance each day. "
        "How many miles did {name} drive per day?",
        lambda a, b: a // b,
        "{a} / {b} = {answer}",
    ),
    (
        "A school has {a} students divided equally into {b} classes. "
        "How many students are in each class?",
        lambda a, b: a // b,
        "{a} / {b} = {answer}",
    ),
]

_TEMPLATES_BY_CATEGORY: dict[str, list] = {
    "addition": _ADDITION_TEMPLATES,
    "subtraction": _SUBTRACTION_TEMPLATES,
    "multiplication": _MULTIPLICATION_TEMPLATES,
    "division": _DIVISION_TEMPLATES,
}

_NAMES = [
    "Emma",
    "Liam",
    "Olivia",
    "Noah",
    "Ava",
    "James",
    "Sophia",
    "Mason",
    "Isabella",
    "Ethan",
    "Mia",
    "Lucas",
    "Harper",
    "Logan",
    "Amelia",
    "Alex",
    "Chloe",
    "Daniel",
    "Lily",
    "Henry",
]

_ITEMS = [
    "apples",
    "books",
    "pencils",
    "cookies",
    "marbles",
    "stickers",
    "balloons",
    "flowers",
    "candies",
    "oranges",
    "cards",
    "shells",
    "stones",
    "buttons",
    "beads",
    "stamps",
    "coins",
    "toys",
    "crayons",
    "cupcakes",
]

# Difficulty ranges: (min_a, max_a, min_b, max_b)
_DIFFICULTY_RANGES: dict[str, tuple[int, int, int, int]] = {
    "easy": (1, 20, 1, 20),
    "medium": (10, 100, 10, 100),
    "hard": (50, 500, 10, 200),
}


def _pick_numbers(
    difficulty: str, operation: str, rng: random.Random
) -> tuple[int, int]:
    """Pick appropriate numbers for a given difficulty and operation."""
    min_a, max_a, min_b, max_b = _DIFFICULTY_RANGES[difficulty]
    a = rng.randint(min_a, max_a)
    b = rng.randint(min_b, max_b)

    if operation == "subtraction":
        # Ensure a > b so the result is non-negative
        if a < b:
            a, b = b, a
        if a == b:
            a += rng.randint(1, 5)

    elif operation == "division":
        # Ensure clean division: pick b first, then make a a multiple of b
        b = rng.randint(max(2, min_b), min(max_b, 20 if difficulty == "easy" else 50))
        quotient = rng.randint(min_a, max_a)
        a = b * quotient

    return a, b


def generate_distractors(
    correct_answer: int,
    num_distractors: int = 3,
    rng: Optional[random.Random] = None,
) -> list[int]:
    """Generate plausible wrong answers for a math problem.

    Creates distractors based on common student errors:
    - Off-by-one errors (correct +/- 1)
    - Magnitude errors (correct +/- 10)
    - Sign/operation confusion (doubling, halving)
    - Digit transposition for larger numbers

    Args:
        correct_answer: The correct numerical answer.
        num_distractors: Number of wrong answers to generate.
        rng: Optional random.Random instance for reproducibility.

    Returns:
        A list of distractor values (all distinct from each other and the
        correct answer).
    """
    if rng is None:
        rng = random.Random()

    candidates: set[int] = set()

    # Off-by-one errors
    candidates.add(correct_answer + 1)
    candidates.add(correct_answer - 1)

    # Off-by-ten errors
    candidates.add(correct_answer + 10)
    candidates.add(correct_answer - 10)

    # Operation confusion
    if correct_answer > 2:
        candidates.add(correct_answer * 2)
        candidates.add(correct_answer // 2)

    # Nearby random values
    for offset in [2, 3, 5, 11, 15, 20]:
        candidates.add(correct_answer + offset)
        candidates.add(correct_answer - offset)

    # Digit transposition for larger numbers
    answer_str = str(abs(correct_answer))
    if len(answer_str) >= 2:
        chars = list(answer_str)
        i = rng.randint(0, len(chars) - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        transposed = int("".join(chars))
        if correct_answer < 0:
            transposed = -transposed
        candidates.add(transposed)

    # Remove the correct answer and any non-positive values if correct > 0
    candidates.discard(correct_answer)
    if correct_answer > 0:
        candidates = {c for c in candidates if c > 0}

    candidate_list = sorted(candidates)
    rng.shuffle(candidate_list)

    if len(candidate_list) < num_distractors:
        # Generate additional random distractors
        magnitude = max(abs(correct_answer), 5)
        while len(candidate_list) < num_distractors:
            offset = rng.randint(1, max(magnitude, 10))
            val = correct_answer + rng.choice([-1, 1]) * offset
            if val != correct_answer and val not in candidate_list:
                if correct_answer > 0 and val <= 0:
                    continue
                candidate_list.append(val)

    return candidate_list[:num_distractors]


def _generate_single_problem(
    operation: str,
    difficulty: str,
    rng: random.Random,
) -> dict[str, Any]:
    """Generate a single math word problem with multiple-choice options."""
    templates = _TEMPLATES_BY_CATEGORY[operation]
    template_text, compute_fn, explanation_tpl = rng.choice(templates)

    a, b = _pick_numbers(difficulty, operation, rng)
    answer = compute_fn(a, b)

    name = rng.choice(_NAMES)
    name2 = rng.choice([n for n in _NAMES if n != name])
    item = rng.choice(_ITEMS)

    question = template_text.format(name=name, name2=name2, a=a, b=b, item=item)
    explanation = explanation_tpl.format(a=a, b=b, answer=answer)

    distractors = generate_distractors(answer, num_distractors=3, rng=rng)
    choices_raw = [answer] + distractors
    rng.shuffle(choices_raw)
    correct_index = choices_raw.index(answer)
    correct_letter = chr(65 + correct_index)

    choices = [str(c) for c in choices_raw]

    return {
        "question": question,
        "choices": choices,
        "answer": correct_letter,
        "explanation": explanation,
        "difficulty": difficulty,
        "category": operation,
    }


def generate_math_problems(
    num_samples: int,
    difficulty: str = "mixed",
    operations: str = "mixed",
    seed: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Generate primary-level math word problems with multiple-choice options.

    Each generated sample contains a realistic word problem about everyday
    scenarios, four answer choices (A-D), the correct answer letter, a
    step-by-step explanation, difficulty level, and category.

    Args:
        num_samples: Number of problems to generate.
        difficulty: One of 'easy', 'medium', 'hard', or 'mixed'.
            When 'mixed', difficulties are sampled uniformly at random.
        operations: One of 'addition', 'subtraction', 'multiplication',
            'division', or 'mixed'. When 'mixed', operations are sampled
            uniformly at random.
        seed: Optional random seed for reproducibility.

    Returns:
        A list of sample dictionaries with keys: question, choices,
        answer, explanation, difficulty, category.

    Raises:
        ValueError: If difficulty or operations are invalid.
    """
    valid_difficulties = {"easy", "medium", "hard", "mixed"}
    valid_operations = set(_TEMPLATES_BY_CATEGORY.keys()) | {"mixed"}

    if difficulty not in valid_difficulties:
        raise ValueError(
            f"Invalid difficulty '{difficulty}'. Must be one of {sorted(valid_difficulties)}"
        )
    if operations not in valid_operations:
        raise ValueError(
            f"Invalid operations '{operations}'. Must be one of {sorted(valid_operations)}"
        )

    rng = random.Random(seed)

    difficulty_choices = (
        list(_DIFFICULTY_RANGES.keys()) if difficulty == "mixed" else [difficulty]
    )
    operation_choices = (
        list(_TEMPLATES_BY_CATEGORY.keys()) if operations == "mixed" else [operations]
    )

    samples: list[dict[str, Any]] = []
    for _ in range(num_samples):
        op = rng.choice(operation_choices)
        diff = rng.choice(difficulty_choices)
        sample = _generate_single_problem(op, diff, rng)
        samples.append(sample)

    logger.info(
        "Generated %d math problems (difficulty=%s, operations=%s)",
        len(samples),
        difficulty,
        operations,
    )
    return samples


def augment_gsm8k(
    dataset: Dataset,
    num_augmented: int,
    seed: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Create variations of GSM8K problems by changing numbers and contexts.

    Takes problems from the GSM8K dataset and produces augmented versions
    by substituting numerical values and contextual nouns/names while
    preserving the underlying mathematical structure.

    Args:
        dataset: A HuggingFace Dataset with 'question' and 'answer' columns,
            as returned by load_gsm8k().
        num_augmented: Number of augmented samples to produce.
        seed: Optional random seed for reproducibility.

    Returns:
        A list of augmented sample dictionaries in the standard format.
    """
    rng = random.Random(seed)

    context_swaps = {
        "apples": _ITEMS,
        "books": _ITEMS,
        "cookies": _ITEMS,
        "pencils": _ITEMS,
        "marbles": _ITEMS,
        "dollars": ["dollars", "euros", "coins", "tokens", "points"],
    }

    samples: list[dict[str, Any]] = []
    dataset_size = len(dataset)

    for _ in range(num_augmented):
        idx = rng.randint(0, dataset_size - 1)
        row = dataset[idx]
        question_text = row["question"]
        answer_text = row["answer"]

        # Extract the final numeric answer from GSM8K format (#### <number>)
        final_answer: Optional[int] = None
        if "####" in answer_text:
            answer_part = answer_text.split("####")[-1].strip()
            answer_part = answer_part.replace(",", "")
            try:
                final_answer = int(float(answer_part))
            except (ValueError, OverflowError):
                final_answer = None

        # Apply name substitutions
        augmented_question = question_text
        for original_name in _NAMES:
            if original_name in augmented_question:
                replacement = rng.choice([n for n in _NAMES if n != original_name])
                augmented_question = augmented_question.replace(
                    original_name, replacement
                )

        # Apply context substitutions
        for original_item, alternatives in context_swaps.items():
            if original_item in augmented_question.lower():
                replacement_item = rng.choice(
                    [alt for alt in alternatives if alt != original_item]
                )
                augmented_question = augmented_question.replace(
                    original_item, replacement_item
                )

        # Apply numerical perturbation: scale numbers by a small factor
        import re

        scale_factor = rng.choice([0.5, 0.75, 1.5, 2.0])
        number_pattern = re.compile(r"\b(\d+)\b")

        def _scale_number(match: re.Match) -> str:
            num = int(match.group(1))
            if num < 2:
                return match.group(0)
            scaled = max(1, int(num * scale_factor))
            return str(scaled)

        augmented_question = number_pattern.sub(_scale_number, augmented_question)

        # Recompute final answer with the same scale factor if available
        if final_answer is not None:
            scaled_answer = max(1, int(final_answer * scale_factor))
        else:
            scaled_answer = rng.randint(1, 100)

        explanation = (
            answer_text.split("####")[0].strip() if "####" in answer_text else ""
        )
        explanation = f"(Adapted from GSM8K) {explanation}"

        distractors = generate_distractors(scaled_answer, num_distractors=3, rng=rng)
        choices_raw = [scaled_answer] + distractors
        rng.shuffle(choices_raw)
        correct_index = choices_raw.index(scaled_answer)
        correct_letter = chr(65 + correct_index)

        sample = {
            "question": augmented_question,
            "choices": [str(c) for c in choices_raw],
            "answer": correct_letter,
            "explanation": explanation,
            "difficulty": "medium",
            "category": "mixed",
        }
        samples.append(sample)

    logger.info(
        "Augmented %d samples from GSM8K dataset (%d source problems)",
        len(samples),
        dataset_size,
    )
    return samples


def generate_samples(num_samples: int, **kwargs: Any) -> list[dict[str, Any]]:
    """Backward-compatible alias for generate_math_problems."""
    return generate_math_problems(num_samples, **kwargs)


def save_samples(samples: list[dict[str, Any]], output: str) -> None:
    """Save generated samples to a JSON file.

    Args:
        samples: List of sample dictionaries to save.
        output: Output file path.
    """
    with open(output, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d samples to %s", len(samples), output)
