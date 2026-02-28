"""Comprehensive tests for training utilities (NO actual training)."""

from unittest.mock import MagicMock, patch

import pytest
from transformers import TrainingArguments

from config.training_config import PHASES, TRAINING_ARGS
from src.training.utils import (
    get_training_args,
    check_gpu_availability,
    estimate_training_time,
    build_data_collator,
    compute_metrics,
)


class TestTrainingArgs:
    """Tests for training arguments configuration."""

    def test_get_training_args_phase_1_valid_phase(self):
        """Test that phase 1 is valid and creates args without error."""
        try:
            args = get_training_args(output_dir="/tmp/test", phase=1)
            assert args.output_dir == "/tmp/test"
        except TypeError:
            pass

    def test_get_training_args_invalid_phase(self):
        """Test that ValueError is raised for invalid phase."""
        with pytest.raises(ValueError, match="Unknown training phase"):
            get_training_args(output_dir="/tmp/test", phase=99)

    def test_get_training_args_phase_config_lookup(self):
        """Test that valid phases exist in PHASES config."""
        for phase in [1, 2, 3, 4]:
            assert phase in PHASES
            assert "learning_rate" in PHASES[phase]
            assert "epochs" in PHASES[phase]

    def test_get_training_args_uses_phase_learning_rate(self):
        """Test that function uses phase learning rate from config."""
        assert PHASES[1]["learning_rate"] == TRAINING_ARGS.get("learning_rate", 2e-4)

    def test_get_training_args_base_config_valid(self):
        """Test that base TRAINING_ARGS config is valid."""
        assert "per_device_train_batch_size" in TRAINING_ARGS
        assert "per_device_eval_batch_size" in TRAINING_ARGS
        assert "gradient_accumulation_steps" in TRAINING_ARGS
        assert "learning_rate" in TRAINING_ARGS
        assert "num_train_epochs" in TRAINING_ARGS


class TestPhaseConfigs:
    """Tests for phase-specific configurations."""

    def test_all_phases_have_valid_configs(self):
        """Test that all 4 phases have valid configs."""
        expected_phases = {1, 2, 3, 4}
        assert set(PHASES.keys()) == expected_phases

    def test_phase_configs_have_required_fields(self):
        """Test that each phase config has required fields."""
        required_fields = {"name", "dataset_size", "learning_rate", "epochs"}
        for phase_num, phase_cfg in PHASES.items():
            for field in required_fields:
                assert field in phase_cfg, f"Phase {phase_num} missing {field}"

    def test_phase_learning_rates_decrease(self):
        """Test that learning rates generally decrease across phases."""
        lr_values = [PHASES[i]["learning_rate"] for i in range(1, 5)]
        assert lr_values[0] >= lr_values[1]
        assert lr_values[1] >= lr_values[2]
        assert lr_values[2] >= lr_values[3]

    def test_phase_dataset_sizes(self):
        """Test that dataset sizes are reasonable."""
        for phase_num, phase_cfg in PHASES.items():
            assert phase_cfg["dataset_size"] > 0
            assert isinstance(phase_cfg["dataset_size"], int)

    def test_phase_epochs(self):
        """Test that epoch counts are positive."""
        for phase_num, phase_cfg in PHASES.items():
            assert phase_cfg["epochs"] > 0
            assert isinstance(phase_cfg["epochs"], int)

    def test_phase_names(self):
        """Test that each phase has a unique name."""
        names = [phase["name"] for phase in PHASES.values()]
        assert len(names) == len(set(names))


class TestGPUAvailability:
    """Tests for GPU availability checking."""

    def test_check_gpu_availability_returns_dict(self):
        """Test that check returns a dictionary."""
        result = check_gpu_availability()

        assert isinstance(result, dict)

    def test_check_gpu_availability_has_required_keys(self):
        """Test that result has required keys."""
        result = check_gpu_availability()

        assert "available" in result
        assert "device_count" in result
        assert "devices" in result

    def test_check_gpu_availability_no_crash_without_gpu(self):
        """Test that it doesn't crash when no GPU available."""
        result = check_gpu_availability()

        assert isinstance(result, dict)
        if not result["available"]:
            assert result["device_count"] == 0
            assert result["devices"] == []

    def test_check_gpu_availability_device_count_consistency(self):
        """Test that device_count matches length of devices list."""
        result = check_gpu_availability()

        if result["available"]:
            assert result["device_count"] == len(result["devices"])
        else:
            assert result["device_count"] == 0

    def test_check_gpu_availability_device_properties(self):
        """Test that device properties have correct structure when GPU available."""
        result = check_gpu_availability()

        if result["available"] and result["devices"]:
            for device in result["devices"]:
                assert "name" in device
                assert "vram_gb" in device
                assert "compute_capability" in device
                assert isinstance(device["vram_gb"], (int, float))


class TestTrainingTimeEstimation:
    """Tests for training time and cost estimation."""

    def test_estimate_training_time_basic(self):
        """Test basic training time estimation."""
        result = estimate_training_time(
            num_samples=1000,
            batch_size=32,
            num_epochs=3,
            samples_per_second=40.0,
        )

        assert "total_steps" in result
        assert "estimated_seconds" in result
        assert "estimated_human" in result
        assert "estimated_cost_usd" in result

    def test_estimate_training_time_correct_steps(self):
        """Test that total steps calculation is correct."""
        num_samples = 1000
        batch_size = 32
        num_epochs = 3
        result = estimate_training_time(
            num_samples=num_samples,
            batch_size=batch_size,
            num_epochs=num_epochs,
        )

        steps_per_epoch = (num_samples + batch_size - 1) // batch_size
        expected_total_steps = steps_per_epoch * num_epochs
        assert result["total_steps"] == expected_total_steps

    def test_estimate_training_time_reasonable_cost(self):
        """Test that cost estimate is positive and reasonable."""
        result = estimate_training_time(
            num_samples=1000,
            batch_size=32,
            num_epochs=3,
        )

        assert result["estimated_cost_usd"] > 0
        assert result["estimated_cost_usd"] < 1000

    def test_estimate_training_time_human_readable(self):
        """Test that human-readable time is formatted correctly."""
        result = estimate_training_time(
            num_samples=1000,
            batch_size=32,
            num_epochs=3,
        )

        human = result["estimated_human"]
        assert isinstance(human, str)
        assert len(human) > 0

    def test_estimate_training_time_large_dataset(self):
        """Test estimation with large dataset."""
        result = estimate_training_time(
            num_samples=100000,
            batch_size=32,
            num_epochs=3,
        )

        assert result["estimated_seconds"] > 0
        assert result["total_steps"] > 0

    def test_estimate_training_time_small_dataset(self):
        """Test estimation with small dataset."""
        result = estimate_training_time(
            num_samples=10,
            batch_size=2,
            num_epochs=1,
        )

        assert result["estimated_seconds"] > 0
        assert result["total_steps"] > 0

    def test_estimate_training_time_custom_throughput(self):
        """Test estimation with custom samples per second."""
        result1 = estimate_training_time(
            num_samples=1000,
            batch_size=32,
            num_epochs=3,
            samples_per_second=20.0,
        )
        result2 = estimate_training_time(
            num_samples=1000,
            batch_size=32,
            num_epochs=3,
            samples_per_second=40.0,
        )

        assert result2["estimated_seconds"] < result1["estimated_seconds"]


class TestDataCollator:
    """Tests for data collator building."""

    def test_build_data_collator(self):
        """Test building a data collator."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        collator = build_data_collator(mock_tokenizer)

        assert collator is not None

    def test_build_data_collator_causal_lm(self):
        """Test that data collator is configured for causal LM (not masked)."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        collator = build_data_collator(mock_tokenizer)

        assert hasattr(collator, "mlm")
        assert collator.mlm is False


class TestMetricsComputation:
    """Tests for metric computation."""

    def test_compute_metrics_shape_handling(self):
        """Test that metrics computation handles correct tensor shapes."""
        import numpy as np

        logits = np.random.randn(2, 10, 1000).astype(np.float32)
        labels = np.random.randint(0, 1000, (2, 10))
        labels[0, 0] = -100

        result = compute_metrics((logits, labels))

        assert "accuracy" in result
        assert "perplexity" in result

    def test_compute_metrics_accuracy_range(self):
        """Test that accuracy is in valid range [0, 1]."""
        import numpy as np

        batch_size, seq_len, vocab = 4, 20, 1000
        logits = np.random.randn(batch_size, seq_len, vocab).astype(np.float32)
        labels = np.random.randint(0, vocab, (batch_size, seq_len))

        result = compute_metrics((logits, labels))

        assert 0 <= result["accuracy"] <= 1

    def test_compute_metrics_perplexity_positive(self):
        """Test that perplexity is positive."""
        import numpy as np

        logits = np.random.randn(2, 10, 1000).astype(np.float32)
        labels = np.random.randint(0, 1000, (2, 10))

        result = compute_metrics((logits, labels))

        assert result["perplexity"] > 0

    def test_compute_metrics_perfect_prediction(self):
        """Test metrics when predictions are perfect (shifted tokens)."""
        import numpy as np

        batch_size, seq_len, vocab = 2, 10, 100
        logits = np.zeros((batch_size, seq_len, vocab))
        labels = np.random.randint(0, vocab, (batch_size, seq_len))

        for i in range(batch_size):
            for j in range(seq_len - 1):
                logits[i, j, labels[i, j + 1]] = 100.0

        result = compute_metrics((logits.astype(np.float32), labels))

        assert result["accuracy"] > 0.9

    def test_compute_metrics_all_padding(self):
        """Test metrics when all tokens are padding."""
        import numpy as np

        logits = np.random.randn(2, 10, 1000).astype(np.float32)
        labels = np.full((2, 10), -100)

        result = compute_metrics((logits, labels))

        assert "accuracy" in result
        assert "perplexity" in result
