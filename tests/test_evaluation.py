"""Comprehensive tests for evaluation utilities."""

from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.evaluator import BaseEvaluator


class ConcreteEvaluator(BaseEvaluator):
    """Concrete implementation of BaseEvaluator for testing."""

    def evaluate(self, dataset):
        """Stub implementation required by abstract base class."""
        return {"results": []}


class TestAnswerExtraction:
    """Tests for answer extraction from model outputs."""

    def test_answer_extraction_numeric_with_hash_format(self):
        """Test extracting numeric answer in #### format (GSM8K style)."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        response = "Let me think... The answer is #### 42"
        extracted = evaluator.extract_answer(response)

        assert extracted == "42"

    def test_answer_extraction_numeric_with_hash_and_spaces(self):
        """Test extracting numeric answer with variable spacing."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        response = "The final result is    ####    123"
        extracted = evaluator.extract_answer(response)

        assert extracted == "123"

    def test_answer_extraction_numeric_with_commas(self):
        """Test extracting numeric answer with thousands separators."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        response = "The answer is #### 1,234,567"
        extracted = evaluator.extract_answer(response)

        assert extracted == "1234567"

    def test_answer_extraction_numeric_with_decimal(self):
        """Test extracting decimal numeric answer."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        response = "The result is #### 3.14159"
        extracted = evaluator.extract_answer(response)

        assert extracted == "3.14159"

    def test_answer_extraction_letter_explicit_format(self):
        """Test extracting letter answer in explicit format."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        response = "The correct answer is A"
        extracted = evaluator.extract_answer(response)

        assert extracted == "A"

    def test_answer_extraction_letter_with_colon(self):
        """Test extracting letter answer with colon separator."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        response = "The answer is: B"
        extracted = evaluator.extract_answer(response)

        assert extracted == "B"

    def test_answer_extraction_letter_lowercase(self):
        """Test extracting letter answer in lowercase (normalized to uppercase)."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        response = "The correct answer is c"
        extracted = evaluator.extract_answer(response)

        assert extracted == "C"

    def test_answer_extraction_correct_answer_text(self):
        """Test extracting answer from 'correct answer is' format."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        response = "Therefore, the correct answer is D"
        extracted = evaluator.extract_answer(response)

        assert extracted == "D"

    def test_answer_extraction_no_match(self):
        """Test that None is returned when no pattern matches."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        response = "I don't know the answer"
        extracted = evaluator.extract_answer(response)

        assert extracted is None

    def test_answer_extraction_multiple_patterns_priority(self):
        """Test that earlier patterns take priority."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        response = "#### 42 and also the answer is A"
        extracted = evaluator.extract_answer(response)

        assert extracted == "42"

    def test_answer_extraction_parenthesis_format(self):
        """Test extracting answer in parenthesis format."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        response = "The options are:\nA) First choice\nB) Second choice"
        extracted = evaluator.extract_answer(response)

        assert extracted == "A"

    def test_answer_extraction_negative_number(self):
        """Test extracting negative numeric answer."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        response = "The result is #### -42"
        extracted = evaluator.extract_answer(response)

        assert extracted == "-42"


class TestAnswerNormalization:
    """Tests for answer normalization."""

    def test_normalize_answer_float_to_int(self):
        """Test normalizing '42.0' -> '42'."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        normalized = evaluator._normalize_answer("42.0")

        assert normalized == "42"

    def test_normalize_answer_multiple_decimal_zeros(self):
        """Test normalizing '3.00' -> '3'."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        normalized = evaluator._normalize_answer("3.00")

        assert normalized == "3"

    def test_normalize_answer_whitespace_stripping(self):
        """Test stripping leading/trailing whitespace."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        normalized = evaluator._normalize_answer("  42  ")

        assert normalized == "42"

    def test_normalize_answer_lowercase_conversion(self):
        """Test converting to lowercase."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        normalized = evaluator._normalize_answer("ANSWER")

        assert normalized == "answer"

    def test_normalize_answer_comma_removal(self):
        """Test removing commas from numbers."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        normalized = evaluator._normalize_answer("1,234,567")

        assert normalized == "1234567"

    def test_normalize_answer_letter(self):
        """Test normalizing single letter answer."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        normalized = evaluator._normalize_answer("B")

        assert normalized == "b"

    def test_normalize_answer_true_decimal(self):
        """Test that true decimals are preserved."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        normalized = evaluator._normalize_answer("3.14")

        assert normalized == "3.14"

    def test_normalize_answer_non_numeric(self):
        """Test normalizing non-numeric answer."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        normalized = evaluator._normalize_answer("Paris")

        assert normalized == "paris"


class TestAnswerExtractionMultipleFormats:
    """Tests for various answer formats."""

    def test_extract_answer_gsm8k_format(self):
        """Test extracting answer from standard GSM8K format."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        response = "Let me work through this step by step.\nFirst, I calculate X.\nThen I get Y.\n#### 100"
        extracted = evaluator.extract_answer(response)

        assert extracted == "100"

    def test_extract_answer_multiple_choice_format(self):
        """Test extracting from multiple choice format."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        response = """
        Options:
        A) First choice
        B) Second choice
        C) Third choice
        D) Fourth choice

        The answer is C
        """
        extracted = evaluator.extract_answer(response)

        assert extracted == "C"

    def test_extract_answer_natural_language_format(self):
        """Test extracting from natural language."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        response = "Based on my analysis, the correct answer is A."
        extracted = evaluator.extract_answer(response)

        assert extracted == "A"

    def test_extract_answer_verbose_response(self):
        """Test extracting from verbose response."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        response = """
        This is a complex problem. Let me break it down:
        1. First consideration
        2. Second consideration
        3. Final calculation

        Therefore, the answer is 42
        """
        extracted = evaluator.extract_answer(response)

        assert extracted == "42"

    def test_extract_answer_case_insensitive(self):
        """Test that extraction is case insensitive."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        responses = [
            "the ANSWER is A",
            "the answer IS a",
            "The Answer Is A",
        ]

        for response in responses:
            extracted = evaluator.extract_answer(response)
            assert extracted == "A"

    def test_extract_answer_with_extra_text(self):
        """Test extracting answer from text with extra surrounding content."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        response = "blah blah #### 999 more text here"
        extracted = evaluator.extract_answer(response)

        assert extracted == "999"


class TestErrorAnalysis:
    """Tests for error categorization."""

    def test_evaluator_initialization(self):
        """Test initializer sets up evaluator correctly."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.eos_token_id = 2

        evaluator = ConcreteEvaluator(
            mock_model,
            mock_tokenizer,
            max_new_tokens=256,
            temperature=0.5,
            batch_size=4,
        )

        assert evaluator.max_new_tokens == 256
        assert evaluator.temperature == 0.5
        assert evaluator.batch_size == 4

    def test_evaluator_pad_token_assignment(self):
        """Test that pad token is assigned from eos if not set."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.eos_token_id = 2

        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        assert mock_tokenizer.pad_token == "<eos>"
        assert mock_tokenizer.pad_token_id == 2

    def test_evaluator_pattern_list_not_empty(self):
        """Test that evaluator has answer patterns defined."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        assert hasattr(evaluator, "ANSWER_PATTERNS")
        assert len(evaluator.ANSWER_PATTERNS) > 0

    def test_evaluator_multiple_answer_patterns(self):
        """Test that multiple answer extraction patterns are defined."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        pattern_count = len(evaluator.ANSWER_PATTERNS)
        assert pattern_count >= 3

    def test_extract_answer_with_various_question_types(self):
        """Test answer extraction across different question types."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        test_cases = [
            ("#### 42", "42"),
            ("the answer is A", "A"),
            ("correct answer is C", "C"),
            ("A)", "A"),
            ("123", "123"),
        ]

        for response, expected in test_cases:
            extracted = evaluator.extract_answer(response)
            assert extracted == expected, f"Failed for response: {response}"


class TestEvaluatorConfiguration:
    """Tests for evaluator configuration and properties."""

    def test_evaluator_default_max_new_tokens(self):
        """Test default max_new_tokens value."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        assert evaluator.max_new_tokens == 512

    def test_evaluator_default_temperature(self):
        """Test default temperature value."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        assert evaluator.temperature == 0.0

    def test_evaluator_default_batch_size(self):
        """Test default batch size value."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        assert evaluator.batch_size == 8

    def test_evaluator_custom_configuration(self):
        """Test creating evaluator with custom configuration."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        evaluator = ConcreteEvaluator(
            mock_model,
            mock_tokenizer,
            max_new_tokens=1024,
            temperature=0.7,
            batch_size=16,
        )

        assert evaluator.max_new_tokens == 1024
        assert evaluator.temperature == 0.7
        assert evaluator.batch_size == 16

    def test_evaluator_stores_model_reference(self):
        """Test that evaluator stores model reference."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        assert evaluator.model is mock_model

    def test_evaluator_stores_tokenizer_reference(self):
        """Test that evaluator stores tokenizer reference."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        evaluator = ConcreteEvaluator(mock_model, mock_tokenizer)

        assert evaluator.tokenizer is mock_tokenizer
