"""Multi-benchmark evaluation runner.

Provides a single entry point for loading a model and running one or more
evaluation benchmarks, then persisting the results.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.gsm8k_evaluator import GSM8KEvaluator

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Loads a model once and runs multiple benchmarks against it.

    Args:
        model_path: Path or Hugging Face Hub identifier for the model to
            evaluate.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

        logger.info("Loading tokenizer from %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        logger.info("Loading model from %s", model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()

    def run_gsm8k(
        self,
        num_samples: Optional[int] = None,
        num_fewshot: int = 5,
    ) -> dict[str, Any]:
        """Run evaluation on the GSM8K benchmark.

        Args:
            num_samples: Number of test examples to evaluate. ``None`` means
                the full test set.
            num_fewshot: Number of few-shot exemplars to include in prompts.

        Returns:
            A results dictionary as returned by
            :meth:`GSM8KEvaluator.evaluate`.
        """
        logger.info(
            "Starting GSM8K evaluation (num_samples=%s, num_fewshot=%d)",
            num_samples,
            num_fewshot,
        )
        evaluator = GSM8KEvaluator(
            model=self.model,
            tokenizer=self.tokenizer,
            num_fewshot=num_fewshot,
        )
        results = evaluator.evaluate(num_samples=num_samples)
        logger.info(
            "GSM8K evaluation complete: accuracy=%.4f (%d/%d)",
            results["overall_accuracy"],
            results["correct"],
            results["total_problems"],
        )
        return results

    def run_all(self) -> dict[str, Any]:
        """Run all registered benchmarks.

        Currently supported benchmarks:
        - GSM8K

        Returns:
            A dictionary mapping benchmark names to their result dicts,
            plus top-level metadata.
        """
        logger.info("Running all benchmarks for model %s", self.model_path)

        gsm8k_results = self.run_gsm8k()

        return {
            "model_path": self.model_path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "benchmarks": {
                "gsm8k": gsm8k_results,
            },
        }

    def save_results(
        self,
        results: dict[str, Any],
        output_dir: str,
    ) -> Path:
        """Save benchmark results to a JSON file.

        The file is named ``eval_results_<timestamp>.json`` inside
        ``output_dir``.

        Args:
            results: The results dictionary to persist.
            output_dir: Directory in which to write the output file.

        Returns:
            The ``Path`` to the written JSON file.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"eval_results_{timestamp}.json"
        filepath = out_path / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info("Results saved to %s", filepath)
        return filepath
