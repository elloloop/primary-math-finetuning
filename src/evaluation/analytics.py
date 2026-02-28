"""Error analysis utilities for evaluation results.

Provides functions to categorize prediction errors, generate structured
error reports, and print human-readable summaries.
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset


def _classify_operation(question: str) -> str:
    """Heuristically classify the dominant arithmetic operation in a question.

    Args:
        question: The problem statement text.

    Returns:
        One of ``"addition"``, ``"subtraction"``, ``"multiplication"``,
        ``"division"``, or ``"mixed"`` if multiple operations are detected.
    """
    operation_signals: dict[str, list[str]] = {
        "addition": [
            "total",
            "sum",
            "combined",
            "altogether",
            "together",
            "plus",
            "more than",
            "added",
            "in all",
        ],
        "subtraction": [
            "difference",
            "left",
            "remain",
            "fewer",
            "less than",
            "minus",
            "spent",
            "gave away",
            "lost",
            "decrease",
        ],
        "multiplication": [
            "times",
            "each",
            "per",
            "every",
            "product",
            "multiplied",
            "twice",
            "triple",
            "double",
        ],
        "division": [
            "divided",
            "split",
            "shared equally",
            "per each",
            "ratio",
            "half",
            "third",
            "quarter",
            "average",
        ],
    }

    lower_q = question.lower()
    detected: list[str] = []

    for operation, keywords in operation_signals.items():
        if any(kw in lower_q for kw in keywords):
            detected.append(operation)

    if len(detected) == 0:
        return "mixed"
    if len(detected) == 1:
        return detected[0]
    return "mixed"


def _estimate_complexity(question: str) -> str:
    """Estimate the reasoning complexity of a math problem.

    Uses sentence count as a proxy for the number of reasoning steps.

    Args:
        question: The problem statement text.

    Returns:
        ``"simple"`` (1--2 sentences), ``"moderate"`` (3--4), or
        ``"complex"`` (5+).
    """
    sentences = [s.strip() for s in re.split(r"[.!?]+", question) if s.strip()]
    count = len(sentences)

    if count <= 2:
        return "simple"
    if count <= 4:
        return "moderate"
    return "complex"


def _classify_error_type(prediction: dict[str, Any]) -> str:
    """Classify the type of error in a wrong prediction.

    Args:
        prediction: A single prediction dict with ``predicted_answer``,
            ``gold_answer``, and ``full_response`` fields.

    Returns:
        One of:
        - ``"extraction_failure"``: Model produced reasoning but the answer
          could not be parsed.
        - ``"calculation_error"``: Answer was extracted but differs from gold.
        - ``"no_output"``: Model generated an empty or trivial response.
        - ``"wrong_reasoning"``: Catch-all for other failures.
    """
    predicted = prediction.get("predicted_answer", "")
    response = prediction.get("full_response", "")

    if not response or response.strip() == "":
        return "no_output"

    if not predicted or predicted.strip() == "":
        # The model produced text but no answer was extractable.
        return "extraction_failure"

    # If the predicted answer is a number and the gold is a number, treat
    # it as a calculation error.
    try:
        float(predicted)
        float(prediction.get("gold_answer", ""))
        return "calculation_error"
    except (ValueError, TypeError):
        pass

    return "wrong_reasoning"


def analyze_errors(
    results: dict[str, Any],
    dataset: Optional[Dataset] = None,
) -> dict[str, Any]:
    """Categorize errors in evaluation results by operation, complexity, and type.

    Args:
        results: The results dict returned by an evaluator's ``evaluate``
            method. Must contain a ``predictions`` list.
        dataset: Optional dataset for additional metadata. Not currently used
            beyond what is already in ``predictions``.

    Returns:
        A dictionary with the following keys:

        - ``total_errors``: Number of incorrect predictions.
        - ``error_rate``: Fraction of predictions that are incorrect.
        - ``by_operation``: Mapping from operation category to error count.
        - ``by_complexity``: Mapping from complexity level to error count.
        - ``by_error_type``: Mapping from error type to error count.
        - ``error_details``: List of dicts, one per error, each containing
          the question, gold/predicted answers, error type, operation, and
          complexity.
    """
    predictions = results.get("predictions", [])
    errors = [p for p in predictions if not p.get("is_correct", False)]

    operation_counter: Counter[str] = Counter()
    complexity_counter: Counter[str] = Counter()
    error_type_counter: Counter[str] = Counter()
    error_details: list[dict[str, Any]] = []

    for pred in errors:
        question = pred.get("question", "")
        operation = _classify_operation(question)
        complexity = _estimate_complexity(question)
        error_type = _classify_error_type(pred)

        operation_counter[operation] += 1
        complexity_counter[complexity] += 1
        error_type_counter[error_type] += 1

        error_details.append(
            {
                "question": question,
                "gold_answer": pred.get("gold_answer", ""),
                "predicted_answer": pred.get("predicted_answer", ""),
                "error_type": error_type,
                "operation": operation,
                "complexity": complexity,
            }
        )

    total = len(predictions)
    return {
        "total_errors": len(errors),
        "error_rate": len(errors) / total if total > 0 else 0.0,
        "by_operation": dict(operation_counter),
        "by_complexity": dict(complexity_counter),
        "by_error_type": dict(error_type_counter),
        "error_details": error_details,
    }


def generate_error_report(
    results: dict[str, Any],
    dataset: Optional[Dataset] = None,
    output_path: Optional[str] = None,
) -> dict[str, Any]:
    """Generate a full JSON error analysis report.

    The report structure matches the project specification format and includes
    aggregate statistics plus detailed per-error breakdowns.

    Args:
        results: The results dict from an evaluator.
        dataset: Optional dataset for additional metadata.
        output_path: If provided, write the report as JSON to this file path.

    Returns:
        The complete report dictionary.
    """
    analysis = analyze_errors(results, dataset)

    report: dict[str, Any] = {
        "summary": {
            "total_problems": results.get("total_problems", 0),
            "correct": results.get("correct", 0),
            "incorrect": results.get("incorrect", 0),
            "overall_accuracy": results.get("overall_accuracy", 0.0),
        },
        "error_analysis": {
            "total_errors": analysis["total_errors"],
            "error_rate": analysis["error_rate"],
            "by_operation": analysis["by_operation"],
            "by_complexity": analysis["by_complexity"],
            "by_error_type": analysis["by_error_type"],
        },
        "error_details": analysis["error_details"],
    }

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

    return report


def print_summary(results: dict[str, Any]) -> None:
    """Print a formatted evaluation summary to stdout.

    Args:
        results: The results dict from an evaluator, expected to contain
            ``overall_accuracy``, ``total_problems``, ``correct``, and
            ``incorrect`` keys.
    """
    accuracy = results.get("overall_accuracy", 0.0)
    total = results.get("total_problems", 0)
    correct = results.get("correct", 0)
    incorrect = results.get("incorrect", 0)

    width = 50
    print("=" * width)
    print("  Evaluation Summary")
    print("=" * width)
    print(f"  Total problems:  {total}")
    print(f"  Correct:         {correct}")
    print(f"  Incorrect:       {incorrect}")
    print(f"  Accuracy:        {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("=" * width)
