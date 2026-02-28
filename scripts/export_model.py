#!/usr/bin/env python
import argparse
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_path", required=True)
    args = p.parse_args()
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(args.model_path, args.output_path, dirs_exist_ok=True)
    print(f"Exported model to {args.output_path}")


if __name__ == "__main__":
    main()
