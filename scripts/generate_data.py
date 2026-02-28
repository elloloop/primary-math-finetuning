#!/usr/bin/env python
import argparse
from src.data.data_generator import generate_samples, save_samples
from src.data.data_validator import validate_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_samples", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--difficulty", default="mixed")
    p.add_argument("--operations", default="mixed")
    p.add_argument("--use_gpt4", action="store_true")
    p.add_argument("--api_key")
    p.add_argument("--augment_gsm8k", action="store_true")
    p.add_argument("--validate", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    data = generate_samples(args.num_samples)
    save_samples(data, args.output)
    if args.validate:
        ok, errs = validate_dataset(args.output)
        print("valid" if ok else f"invalid: {errs[:5]}")
    print(f"saved {len(data)} samples to {args.output}")


if __name__ == "__main__":
    main()
