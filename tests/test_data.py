"""Comprehensive tests for src/data module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from src.data.data_formatter import format_for_training, format_batch, to_chatml
from src.data.data_generator import (
    generate_math_problems,
    generate_distractors,
)
from src.data.data_loader import (
    load_json_dataset,
    create_train_val_split,
)
from src.data.data_validator import (
    validate_sample,
    validate_dataset,
    detect_duplicates,
    get_dataset_stats,
)


class TestDataLoading:
    """Tests for data loading functionality."""

    def test_load_json_dataset_from_json_file(self, tmp_path):
        """Test loading a JSON file with list of records."""
        samples = [
            {
                "question": "What is 2+2?",
                "choices": ["3", "4", "5", "6"],
                "answer": "B",
            },
            {
                "question": "What is 3+3?",
                "choices": ["5", "6", "7", "8"],
                "answer": "B",
            },
        ]
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps(samples))

        loaded = load_json_dataset(json_file)
        assert len(loaded) == 2
        assert loaded[0]["question"] == "What is 2+2?"
        assert loaded[1]["question"] == "What is 3+3?"

    def test_load_json_dataset_from_jsonl_file(self, tmp_path):
        """Test loading a JSONL file (one JSON object per line)."""
        jsonl_file = tmp_path / "data.jsonl"
        lines = [
            json.dumps(
                {
                    "question": "What is 2+2?",
                    "choices": ["3", "4", "5", "6"],
                    "answer": "B",
                }
            ),
            json.dumps(
                {
                    "question": "What is 3+3?",
                    "choices": ["5", "6", "7", "8"],
                    "answer": "B",
                }
            ),
        ]
        jsonl_file.write_text("\n".join(lines))

        loaded = load_json_dataset(jsonl_file)
        assert len(loaded) == 2
        assert loaded[0]["question"] == "What is 2+2?"
        assert loaded[1]["question"] == "What is 3+3?"

    def test_load_json_dataset_file_not_found(self):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            load_json_dataset("/nonexistent/path/file.json")

    def test_load_json_dataset_unsupported_format(self, tmp_path):
        """Test that ValueError is raised for unsupported file formats."""
        unsupported_file = tmp_path / "data.txt"
        unsupported_file.write_text("some content")

        with pytest.raises(ValueError, match="Unsupported file format"):
            load_json_dataset(unsupported_file)

    def test_load_json_dataset_invalid_json(self, tmp_path):
        """Test that ValueError is raised for invalid JSON."""
        json_file = tmp_path / "data.json"
        json_file.write_text("not valid json {[}")

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_json_dataset(json_file)

    def test_load_json_dataset_single_json_object(self, tmp_path):
        """Test loading a JSON file with a single object."""
        sample = {
            "question": "What is 2+2?",
            "choices": ["3", "4", "5", "6"],
            "answer": "B",
        }
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps(sample))

        loaded = load_json_dataset(json_file)
        assert len(loaded) == 1
        assert loaded[0]["question"] == "What is 2+2?"


class TestDataFormatting:
    """Tests for ChatML format conversion."""

    def test_format_qwen_chat_basic(self):
        """Test ChatML formatting produces correct template with im_start/im_end tags."""
        record = {
            "question": "What is 1+1?",
            "choices": ["1", "2", "3", "4"],
            "answer": "B",
            "explanation": "Two equals two",
        }
        result = format_for_training(record)

        assert "text" in result
        text = result["text"]
        assert "<|im_start|>system" in text
        assert "<|im_end|>" in text
        assert "<|im_start|>user" in text
        assert "<|im_start|>assistant" in text
        assert "The correct answer is: B" in text
        assert "Options:" in text
        assert "A) 1" in text
        assert "B) 2" in text
        assert "C) 3" in text
        assert "D) 4" in text

    def test_format_qwen_chat_custom_system_message(self):
        """Test ChatML formatting with custom system message."""
        record = {
            "question": "What is 1+1?",
            "choices": ["1", "2", "3", "4"],
            "answer": "B",
            "explanation": "Two equals two",
        }
        custom_system = "You are a test assistant."
        result = format_for_training(record, system_message=custom_system)

        assert custom_system in result["text"]

    def test_format_qwen_chat_missing_explanation(self):
        """Test ChatML formatting when explanation is missing."""
        record = {
            "question": "What is 1+1?",
            "choices": ["1", "2", "3", "4"],
            "answer": "B",
        }
        result = format_for_training(record)

        assert "text" in result
        assert "The correct answer is: B" in result["text"]

    def test_format_qwen_chat_wrong_choice_count(self):
        """Test that ValueError is raised when choices != 4."""
        record = {
            "question": "What is 1+1?",
            "choices": ["1", "2", "3"],
            "answer": "B",
        }
        with pytest.raises(ValueError, match="exactly 4 choices"):
            format_for_training(record)

    def test_format_qwen_chat_missing_required_field(self):
        """Test that KeyError is raised for missing required fields."""
        record = {
            "question": "What is 1+1?",
            "choices": ["1", "2", "3", "4"],
        }
        with pytest.raises(KeyError):
            format_for_training(record)

    def test_format_batch(self):
        """Test batch formatting of multiple samples."""
        samples = [
            {
                "question": "What is 1+1?",
                "choices": ["1", "2", "3", "4"],
                "answer": "B",
            },
            {
                "question": "What is 2+2?",
                "choices": ["3", "4", "5", "6"],
                "answer": "B",
            },
        ]
        results = format_batch(samples)

        assert len(results) == 2
        assert all("text" in r for r in results)
        assert "The correct answer is: B" in results[0]["text"]
        assert "The correct answer is: B" in results[1]["text"]

    def test_to_chatml_convenience_wrapper(self):
        """Test to_chatml convenience wrapper returns string not dict."""
        record = {
            "question": "What is 1+1?",
            "choices": ["1", "2", "3", "4"],
            "answer": "B",
        }
        result = to_chatml(record)

        assert isinstance(result, str)
        assert "<|im_start|>system" in result
        assert "The correct answer is: B" in result


class TestDataValidation:
    """Tests for sample and dataset validation."""

    def test_data_validation_valid_sample(self):
        """Test validation of a correctly formatted sample."""
        sample = {
            "question": "What is 1+1?",
            "choices": ["1", "2", "3", "4"],
            "answer": "B",
        }
        errors = validate_sample(sample)

        assert errors == []

    def test_data_validation_invalid_answer(self):
        """Test validation catches answer not in A-D."""
        sample = {
            "question": "What is 1+1?",
            "choices": ["1", "2", "3", "4"],
            "answer": "E",
        }
        errors = validate_sample(sample)

        assert len(errors) > 0
        assert any("answer" in err.lower() for err in errors)

    def test_data_validation_wrong_choice_count(self):
        """Test validation catches != 4 choices."""
        sample = {
            "question": "What is 1+1?",
            "choices": ["1", "2", "3"],
            "answer": "B",
        }
        errors = validate_sample(sample)

        assert len(errors) > 0
        assert any("choices" in err.lower() for err in errors)

    def test_data_validation_missing_required_field(self):
        """Test validation catches missing required fields."""
        sample = {
            "question": "What is 1+1?",
            "choices": ["1", "2", "3", "4"],
        }
        errors = validate_sample(sample)

        assert len(errors) > 0
        assert any("answer" in err.lower() for err in errors)

    def test_data_validation_non_unique_choices(self):
        """Test validation catches non-unique choices."""
        sample = {
            "question": "What is 1+1?",
            "choices": ["1", "2", "2", "3"],
            "answer": "B",
        }
        errors = validate_sample(sample)

        assert len(errors) > 0
        assert any("unique" in err.lower() for err in errors)

    def test_data_validation_empty_question(self):
        """Test validation catches empty question."""
        sample = {
            "question": "",
            "choices": ["1", "2", "3", "4"],
            "answer": "B",
        }
        errors = validate_sample(sample)

        assert len(errors) > 0
        assert any("question" in err.lower() for err in errors)

    def test_dataset_validation_valid(self, tmp_path):
        """Test dataset validation with valid file."""
        samples = [
            {
                "question": "What is 1+1?",
                "choices": ["1", "2", "3", "4"],
                "answer": "B",
            },
            {
                "question": "What is 2+2?",
                "choices": ["3", "4", "5", "6"],
                "answer": "B",
            },
        ]
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps(samples))

        is_valid, errors = validate_dataset(str(json_file))

        assert is_valid
        assert len(errors) == 0

    def test_dataset_validation_with_list(self):
        """Test dataset validation with list directly."""
        samples = [
            {
                "question": "What is 1+1?",
                "choices": ["1", "2", "3", "4"],
                "answer": "B",
            },
        ]
        is_valid, errors = validate_dataset(samples)

        assert is_valid
        assert len(errors) == 0

    def test_dataset_validation_with_invalid_samples(self):
        """Test dataset validation catches invalid samples."""
        samples = [
            {
                "question": "What is 1+1?",
                "choices": ["1", "2", "3"],
                "answer": "B",
            },
        ]
        is_valid, errors = validate_dataset(samples)

        assert not is_valid
        assert len(errors) > 0


class TestDuplicateDetection:
    """Tests for duplicate detection."""

    def test_detect_duplicates_with_exact_duplicate(self):
        """Test detection of exact duplicate questions."""
        samples = [
            {
                "question": "What is 1+1?",
                "choices": ["1", "2", "3", "4"],
                "answer": "B",
            },
            {
                "question": "What is 1+1?",
                "choices": ["1", "2", "3", "4"],
                "answer": "B",
            },
            {
                "question": "What is 2+2?",
                "choices": ["3", "4", "5", "6"],
                "answer": "B",
            },
        ]
        duplicates = detect_duplicates(samples, similarity_threshold=0.85)

        assert len(duplicates) > 0
        assert any(pair[2] >= 0.99 for pair in duplicates)

    def test_detect_duplicates_with_no_duplicates(self):
        """Test with completely different questions."""
        samples = [
            {
                "question": "What is 1+1?",
                "choices": ["1", "2", "3", "4"],
                "answer": "B",
            },
            {
                "question": "What is the capital of France?",
                "choices": ["London", "Paris", "Berlin", "Madrid"],
                "answer": "B",
            },
        ]
        duplicates = detect_duplicates(samples, similarity_threshold=0.85)

        assert len(duplicates) == 0

    def test_detect_duplicates_returns_sorted_by_similarity(self):
        """Test that duplicates are sorted by similarity descending."""
        samples = [
            {
                "question": "What is 1+1?",
                "choices": ["1", "2", "3", "4"],
                "answer": "B",
            },
            {
                "question": "What is 1+1?",
                "choices": ["1", "2", "3", "4"],
                "answer": "B",
            },
            {
                "question": "What is one plus one?",
                "choices": ["1", "2", "3", "4"],
                "answer": "B",
            },
        ]
        duplicates = detect_duplicates(samples, similarity_threshold=0.70)

        if len(duplicates) > 1:
            similarities = [d[2] for d in duplicates]
            assert similarities == sorted(similarities, reverse=True)


class TestDatasetStats:
    """Tests for dataset statistics computation."""

    def test_dataset_stats_basic(self):
        """Test dataset statistics computation."""
        samples = [
            {
                "question": "What is 1+1?",
                "choices": ["1", "2", "3", "4"],
                "answer": "A",
                "difficulty": "easy",
                "category": "addition",
                "explanation": "This is easy",
            },
            {
                "question": "What is 2+2?",
                "choices": ["3", "4", "5", "6"],
                "answer": "B",
                "difficulty": "easy",
                "category": "addition",
            },
        ]
        stats = get_dataset_stats(samples)

        assert stats["total_samples"] == 2
        assert "by_difficulty" in stats
        assert "by_category" in stats
        assert "by_answer" in stats
        assert "avg_question_length" in stats
        assert "avg_choices_length" in stats
        assert "has_explanation" in stats

    def test_dataset_stats_empty(self):
        """Test dataset statistics with empty dataset."""
        stats = get_dataset_stats([])

        assert stats["total_samples"] == 0
        assert stats["by_difficulty"] == {}
        assert stats["by_category"] == {}
        assert stats["by_answer"] == {}
        assert stats["avg_question_length"] == 0.0

    def test_dataset_stats_with_missing_fields(self):
        """Test dataset statistics tracks missing optional fields."""
        samples = [
            {
                "question": "What is 1+1?",
                "choices": ["1", "2", "3", "4"],
                "answer": "A",
            },
            {
                "question": "What is 2+2?",
                "choices": ["3", "4", "5", "6"],
                "answer": "B",
                "difficulty": "easy",
            },
        ]
        stats = get_dataset_stats(samples)

        assert "missing_fields" in stats


class TestTrainValSplit:
    """Tests for train/validation split functionality."""

    def test_train_val_split_with_list(self):
        """Test train/val split with a list of dicts."""
        samples = [
            {
                "question": f"What is {i}+{i}?",
                "choices": ["1", "2", "3", "4"],
                "answer": "B",
            }
            for i in range(100)
        ]
        split = create_train_val_split(samples, val_ratio=0.2, seed=42)

        assert "train" in split
        assert "validation" in split
        assert len(split["train"]) + len(split["validation"]) == 100
        assert len(split["validation"]) == 20
        assert len(split["train"]) == 80

    def test_train_val_split_with_dataset(self):
        """Test train/val split with a HuggingFace Dataset."""
        samples = [
            {
                "question": f"What is {i}+{i}?",
                "choices": ["1", "2", "3", "4"],
                "answer": "B",
            }
            for i in range(100)
        ]
        dataset = Dataset.from_list(samples)
        split = create_train_val_split(dataset, val_ratio=0.1, seed=42)

        assert len(split["train"]) + len(split["validation"]) == 100
        assert len(split["validation"]) == 10

    def test_train_val_split_reproducible(self):
        """Test that split is reproducible with same seed."""
        samples = [
            {
                "question": f"What is {i}+{i}?",
                "choices": ["1", "2", "3", "4"],
                "answer": "B",
            }
            for i in range(100)
        ]
        split1 = create_train_val_split(samples, val_ratio=0.2, seed=42)
        split2 = create_train_val_split(samples, val_ratio=0.2, seed=42)

        assert set(split1["train"]["question"]) == set(split2["train"]["question"])

    def test_train_val_split_invalid_ratio_too_small(self):
        """Test that ValueError is raised for val_ratio too small."""
        samples = [{"question": str(i), "choices": ["1", "2", "3", "4"], "answer": "B"} for i in range(100)]

        with pytest.raises(ValueError, match="val_ratio"):
            create_train_val_split(samples, val_ratio=0.0)

    def test_train_val_split_invalid_ratio_too_large(self):
        """Test that ValueError is raised for val_ratio too large."""
        samples = [{"question": str(i), "choices": ["1", "2", "3", "4"], "answer": "B"} for i in range(100)]

        with pytest.raises(ValueError, match="val_ratio"):
            create_train_val_split(samples, val_ratio=1.0)

    def test_train_val_split_empty_dataset(self):
        """Test that ValueError is raised for empty dataset."""
        with pytest.raises(ValueError, match="empty"):
            create_train_val_split([], val_ratio=0.1)


class TestMathProblemGeneration:
    """Tests for synthetic math problem generation."""

    def test_generate_math_problems_basic(self):
        """Test generating 10 problems verifies they have all required fields."""
        problems = generate_math_problems(10, difficulty="easy", operations="addition", seed=42)

        assert len(problems) == 10
        for problem in problems:
            assert "question" in problem
            assert "choices" in problem
            assert "answer" in problem
            assert "explanation" in problem
            assert "difficulty" in problem
            assert "category" in problem
            assert len(problem["choices"]) == 4
            assert problem["answer"] in ["A", "B", "C", "D"]
            assert isinstance(problem["question"], str)
            assert len(problem["question"]) > 0

    def test_generate_math_problems_all_difficulties(self):
        """Test generating problems of each difficulty level."""
        for difficulty in ["easy", "medium", "hard"]:
            problems = generate_math_problems(5, difficulty=difficulty, seed=42)

            assert len(problems) == 5
            assert all(p["difficulty"] == difficulty for p in problems)

    def test_generate_math_problems_all_operations(self):
        """Test generating problems for each operation."""
        for operation in ["addition", "subtraction", "multiplication", "division"]:
            problems = generate_math_problems(5, difficulty="easy", operations=operation, seed=42)

            assert len(problems) == 5
            assert all(p["category"] == operation for p in problems)

    def test_generate_math_problems_mixed(self):
        """Test generating mixed difficulty and operations."""
        problems = generate_math_problems(20, difficulty="mixed", operations="mixed", seed=42)

        assert len(problems) == 20

    def test_generate_math_problems_invalid_difficulty(self):
        """Test that ValueError is raised for invalid difficulty."""
        with pytest.raises(ValueError, match="Invalid difficulty"):
            generate_math_problems(10, difficulty="invalid", operations="addition")

    def test_generate_math_problems_invalid_operation(self):
        """Test that ValueError is raised for invalid operation."""
        with pytest.raises(ValueError, match="Invalid operations"):
            generate_math_problems(10, difficulty="easy", operations="invalid")

    def test_generate_math_problems_reproducible(self):
        """Test that generation is reproducible with same seed."""
        problems1 = generate_math_problems(10, difficulty="easy", operations="addition", seed=42)
        problems2 = generate_math_problems(10, difficulty="easy", operations="addition", seed=42)

        assert [p["question"] for p in problems1] == [p["question"] for p in problems2]
        assert [p["answer"] for p in problems1] == [p["answer"] for p in problems2]


class TestDistractorGeneration:
    """Tests for distractor generation."""

    def test_generate_distractors_basic(self):
        """Test generating 3 distractors different from correct answer."""
        correct = 42
        distractors = generate_distractors(correct, num_distractors=3, rng=None)

        assert len(distractors) == 3
        assert all(d != correct for d in distractors)
        assert len(set(distractors)) == 3

    def test_generate_distractors_multiple_calls(self):
        """Test generating distractors multiple times."""
        correct = 100
        all_distractors = []
        for _ in range(5):
            distractors = generate_distractors(correct, num_distractors=3)
            all_distractors.extend(distractors)
            assert all(d != correct for d in distractors)

    def test_generate_distractors_various_answers(self):
        """Test with various correct answers."""
        for correct in [1, 10, 100, 1000]:
            distractors = generate_distractors(correct, num_distractors=3)

            assert len(distractors) == 3
            assert all(d != correct for d in distractors)
            assert len(set(distractors)) == 3

    def test_generate_distractors_reproducible(self):
        """Test that generation is reproducible with same seed."""
        import random

        rng1 = random.Random(42)
        rng2 = random.Random(42)
        distractors1 = generate_distractors(100, num_distractors=3, rng=rng1)
        distractors2 = generate_distractors(100, num_distractors=3, rng=rng2)

        assert distractors1 == distractors2
