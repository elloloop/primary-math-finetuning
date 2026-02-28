import json
from pathlib import Path
from src.evaluation.gsm8k_evaluator import GSM8KEvaluator


def run_gsm8k_stub(output_dir: str) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    evaluator = GSM8KEvaluator()
    # Placeholder deterministic stub until full lm-eval integration.
    preds = ["A", "B", "C", "D"]
    labels = ["A", "B", "C", "A"]
    acc = evaluator.accuracy(preds, labels)
    result = {"benchmark": "gsm8k", "accuracy": acc, "samples": len(labels)}
    with open(f"{output_dir}/results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result
