#!/usr/bin/env python
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_a", required=True)
    p.add_argument("--checkpoint_b", required=True)
    args = p.parse_args()
    print(f"Compare manually: {args.checkpoint_a} vs {args.checkpoint_b}")


if __name__ == "__main__":
    main()
