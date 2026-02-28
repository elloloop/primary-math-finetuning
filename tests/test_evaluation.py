from src.evaluation.evaluator import BaseEvaluator
from src.evaluation.gsm8k_evaluator import GSM8KEvaluator


def test_answer_extraction():
    e = BaseEvaluator()
    assert e.extract_answer("The correct answer is A") == "A"


def test_gsm8k_evaluation():
    g = GSM8KEvaluator()
    assert g.accuracy(["A", "B"], ["A", "C"]) == 50.0
