"""GSM8K-specific evaluator for grade-school math word problems."""

import re
from typing import Any, Optional

from datasets import Dataset, load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.evaluation.evaluator import BaseEvaluator

# Hand-picked few-shot examples from GSM8K training set that cover a range of
# arithmetic operations and multi-step reasoning.
_FEWSHOT_EXAMPLES: list[dict[str, str]] = [
    {
        "question": (
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast "
            "every morning and bakes muffins for her friends every day with four. "
            "She sells the remainder at the farmers' market daily for $2 per fresh "
            "duck egg. How much in dollars does she make every day at the farmers' market?"
        ),
        "answer": (
            "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\n"
            "She makes 9 * 2 = <<9*2=18>>$18 every day at the farmer's market.\n"
            "#### 18"
        ),
    },
    {
        "question": (
            "A robe takes 2 bolts of blue fiber and half that much white fiber. "
            "How many bolts in total does it take?"
        ),
        "answer": (
            "It takes 2 / 2 = <<2/2=1>>1 bolt of white fiber.\n"
            "So the total bolts needed is 2 + 1 = <<2+1=3>>3.\n"
            "#### 3"
        ),
    },
    {
        "question": (
            "Josh decides to try flipping a house. He buys a house for $80,000 "
            "and then puts in $50,000 in repairs. This increased the value of the "
            "house by 150%. How much profit did he make?"
        ),
        "answer": (
            "The cost of the house and repairs came out to "
            "80,000 + 50,000 = <<80000+50000=130000>>$130,000.\n"
            "He increased the value of the house by "
            "80,000 * 150% = <<80000*1.5=120000>>$120,000.\n"
            "So the new value of the house is "
            "80,000 + 120,000 = <<80000+120000=200000>>$200,000.\n"
            "So he made a profit of "
            "200,000 - 130,000 = <<200000-130000=70000>>$70,000.\n"
            "#### 70000"
        ),
    },
    {
        "question": (
            "Every day, Wendi feeds each of her chickens three cups of mixed "
            "chicken feed, containing a blend of wheat, oats, and dried corn. "
            "She gives the chickens their feed in three separate meals. In the "
            "morning, she gives her flock of chickens 15 cups of feed. In the "
            "afternoon, she gives her chickens another 25 cups of feed. How many "
            "cups of feed does she need to give her chickens in the final meal "
            "of the day if the carry-over from prior meals was 35 cups?"
        ),
        "answer": (
            "If each chicken eats 3 cups of feed per day, and Wendi has given "
            "them 15 + 25 = <<15+25=40>>40 cups so far, with a 35-cup carry-over "
            "making 40 + 35 = <<40+35=75>>75 total cups, and she has "
            "75 / 3 = <<75/3=25>>25 chickens, then 25 * 3 = <<25*3=75>>75 cups "
            "total are needed per day. She still needs to give "
            "75 - 75 = <<75-75=0>>0 cups in the final meal.\n"
            "#### 0"
        ),
    },
    {
        "question": (
            "Kylar went to the store to get his decor items. It cost $250 for "
            "a cabinet, $35 for a table, and $75 for a bed frame. If the store "
            "offered a 15% discount on all items, how much did Kylar pay in total?"
        ),
        "answer": (
            "The total cost before discount is "
            "250 + 35 + 75 = <<250+35+75=360>>$360.\n"
            "The discount is 360 * 15 / 100 = <<360*15/100=54>>$54.\n"
            "So Kylar paid 360 - 54 = <<360-54=306>>$306.\n"
            "#### 306"
        ),
    },
]


class GSM8KEvaluator(BaseEvaluator):
    """Evaluator for the GSM8K grade-school math benchmark.

    Supports few-shot prompting with chain-of-thought reasoning. By default
    uses 5 exemplars drawn from the GSM8K training split.

    Args:
        model: A HuggingFace pretrained language model.
        tokenizer: The tokenizer corresponding to ``model``.
        num_fewshot: Number of few-shot examples to include in the prompt.
        use_cot: Whether to include chain-of-thought reasoning in exemplars.
        **kwargs: Additional keyword arguments forwarded to ``BaseEvaluator``.
    """

    _GSM8K_ANSWER_PATTERN: re.Pattern[str] = re.compile(r"####\s*(-?\d[\d,]*\.?\d*)")

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        num_fewshot: int = 5,
        use_cot: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, tokenizer, **kwargs)
        self.num_fewshot = num_fewshot
        self.use_cot = use_cot

    def _build_fewshot_prompt(self) -> str:
        """Build a few-shot prompt string from GSM8K exemplars.

        When ``use_cot`` is True the full chain-of-thought answer is included;
        otherwise only the final ``#### <number>`` line is shown.

        Returns:
            A formatted string containing ``num_fewshot`` question/answer
            exemplars.
        """
        lines: list[str] = []
        examples = _FEWSHOT_EXAMPLES[: self.num_fewshot]

        for example in examples:
            lines.append(f"Question: {example['question']}")
            if self.use_cot:
                lines.append(f"Answer: {example['answer']}")
            else:
                # Extract only the final numeric answer.
                match = self._GSM8K_ANSWER_PATTERN.search(example["answer"])
                final_answer = match.group(1) if match else ""
                lines.append(f"Answer: #### {final_answer}")
            lines.append("")

        return "\n".join(lines)

    def _extract_gsm8k_answer(self, text: str) -> Optional[str]:
        """Extract the numeric answer following the ``####`` marker.

        This is the canonical GSM8K answer format. The extracted number has
        commas removed and trailing ``.0`` simplified.

        Args:
            text: Model-generated response text.

        Returns:
            The extracted numeric answer as a string, or ``None`` if the
            ``####`` marker is not found.
        """
        matches = self._GSM8K_ANSWER_PATTERN.findall(text)
        if not matches:
            return None
        raw = matches[-1].replace(",", "")
        return self._normalize_answer(raw)

    def evaluate(
        self,
        dataset: Optional[Dataset] = None,
        num_samples: Optional[int] = None,
    ) -> dict[str, Any]:
        """Evaluate the model on the GSM8K test set.

        If no dataset is provided, the ``gsm8k`` ``main`` test split is loaded
        automatically from the Hugging Face Hub.

        Args:
            dataset: An optional pre-loaded GSM8K dataset. Expected to have
                ``question`` and ``answer`` columns.
            num_samples: Limit evaluation to the first ``num_samples`` examples.
                Use ``None`` to evaluate the full test set.

        Returns:
            A dictionary containing:
            - ``overall_accuracy``: Fraction of correct answers (0.0--1.0).
            - ``total_problems``: Number of problems evaluated.
            - ``correct``: Number of correct predictions.
            - ``incorrect``: Number of incorrect predictions.
            - ``predictions``: List of per-example result dicts, each with
              ``question``, ``gold_answer``, ``predicted_answer``,
              ``full_response``, and ``is_correct`` fields.
        """
        if dataset is None:
            ds = load_dataset("gsm8k", "main", split="test")
        else:
            ds = dataset

        if num_samples is not None:
            ds = ds.select(range(min(num_samples, len(ds))))

        fewshot_prefix = self._build_fewshot_prompt()

        # Build prompts for every example.
        prompts: list[str] = []
        gold_answers: list[str] = []

        for example in ds:
            prompt = f"{fewshot_prefix}Question: {example['question']}\nAnswer:"
            prompts.append(prompt)

            gold = self._extract_gsm8k_answer(example["answer"])
            gold_answers.append(gold if gold is not None else "")

        # Generate model responses.
        responses = self.generate(prompts)

        # Score predictions.
        predictions: list[dict[str, Any]] = []
        correct_count = 0

        for i, response in enumerate(responses):
            predicted = self._extract_gsm8k_answer(response)
            predicted_normalized = (
                self._normalize_answer(predicted) if predicted is not None else ""
            )
            gold_normalized = gold_answers[i]
            is_correct = predicted_normalized == gold_normalized

            if is_correct:
                correct_count += 1

            predictions.append(
                {
                    "question": ds[i]["question"],
                    "gold_answer": gold_normalized,
                    "predicted_answer": predicted_normalized,
                    "full_response": response,
                    "is_correct": is_correct,
                }
            )

        total = len(predictions)
        accuracy = correct_count / total if total > 0 else 0.0

        return {
            "overall_accuracy": accuracy,
            "total_problems": total,
            "correct": correct_count,
            "incorrect": total - correct_count,
            "predictions": predictions,
        }
