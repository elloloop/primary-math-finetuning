"""Model configuration values."""

MODEL_CONFIG = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "max_seq_length": 2048,
    "bf16": True,
}

LORA_CONFIG = {
    "r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}
