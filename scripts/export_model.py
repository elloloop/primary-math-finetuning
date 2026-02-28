#!/usr/bin/env python
"""Export a fine-tuned model by merging LoRA weights and saving.

Supports:
    - Merging LoRA adapter weights into the base model.
    - Saving in standard HuggingFace format.
    - Optionally pushing the merged model to HuggingFace Hub.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.lora_config import merge_and_save

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export a fine-tuned model (merge LoRA and save).",
    )
    p.add_argument(
        "--model_path",
        required=True,
        help="Path to the fine-tuned model or LoRA adapter directory.",
    )
    p.add_argument(
        "--output_path",
        required=True,
        help="Directory to save the merged model.",
    )
    p.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the merged model to HuggingFace Hub after saving.",
    )
    p.add_argument(
        "--hub_repo_id",
        default=None,
        help="HuggingFace Hub repository ID (e.g. 'username/model-name'). "
        "Required if --push_to_hub is set.",
    )
    p.add_argument(
        "--hub_token",
        default=None,
        help="HuggingFace Hub token. If not provided, uses cached credentials.",
    )
    p.add_argument(
        "--no_merge",
        action="store_true",
        help="Save adapter weights without merging into the base model.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.push_to_hub and not args.hub_repo_id:
        logger.error("--hub_repo_id is required when --push_to_hub is set.")
        return 1

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # ---- Load model and tokenizer ---------------------------------------
    logger.info("Loading model from %s", args.model_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Try loading as a PEFT model first; fall back to standard model.
    is_peft = False
    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        is_peft = True
        logger.info("Loaded model as PEFT model with LoRA adapters.")
    except Exception:
        logger.info("Not a PEFT checkpoint; loading as standard model.")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    # ---- Merge LoRA weights (if applicable) -----------------------------
    if is_peft and not args.no_merge:
        logger.info("Merging LoRA adapter weights into base model...")
        merge_and_save(model, tokenizer, str(output_path))
        logger.info("Merged model saved to %s", output_path)
    elif is_peft and args.no_merge:
        logger.info("Saving adapter weights without merging...")
        model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        logger.info("Adapter-only model saved to %s", output_path)
    else:
        logger.info("Model is already a standard (non-PEFT) model. Saving as-is.")
        model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        logger.info("Model saved to %s", output_path)

    # ---- Push to Hub ----------------------------------------------------
    if args.push_to_hub:
        logger.info("Pushing model to HuggingFace Hub: %s", args.hub_repo_id)
        try:
            # Re-load the saved merged model for pushing
            push_model = AutoModelForCausalLM.from_pretrained(
                str(output_path),
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            push_model.push_to_hub(
                args.hub_repo_id,
                token=args.hub_token,
                private=True,
            )
            tokenizer.push_to_hub(
                args.hub_repo_id,
                token=args.hub_token,
                private=True,
            )
            logger.info("Model pushed to https://huggingface.co/%s", args.hub_repo_id)
        except Exception as exc:
            logger.error("Failed to push to HuggingFace Hub: %s", exc)
            return 1

    print(f"\nExported model to {output_path}")
    if args.push_to_hub:
        print(f"Pushed to https://huggingface.co/{args.hub_repo_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
