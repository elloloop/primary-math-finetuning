from src.evaluation.evaluator import BaseEvaluator


class GSM8KEvaluator(BaseEvaluator):
    def accuracy(self, predictions: list[str], labels: list[str]) -> float:
        if not labels:
            return 0.0
        correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        return (correct / len(labels)) * 100.0
