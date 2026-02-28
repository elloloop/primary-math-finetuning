#!/usr/bin/env python
from pathlib import Path
from src.data.data_generator import generate_samples, save_samples
from src.evaluation.benchmark_runner import run_gsm8k_stub


def main():
    Path("outputs/results").mkdir(parents=True, exist_ok=True)
    sample_path = "data/examples/quick_test.json"
    save_samples(generate_samples(100), sample_path)
    result = run_gsm8k_stub("outputs/results")
    print(f"Quick test complete. stub accuracy={result['accuracy']:.2f}%")


if __name__ == "__main__":
    main()
