"""Evaluation configuration."""

GSM8K_EVAL_CONFIG = {
    "num_fewshot": 5,
    "use_cot": True,
    "max_new_tokens": 512,
    "temperature": 0.0,
    "batch_size": 8,
    "save_predictions": True,
    "error_analysis": True,
}
