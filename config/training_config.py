"""Training and phase configs."""

TRAINING_ARGS = {
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 0.3,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "num_train_epochs": 3,
    "bf16": True,
    "fp16": False,
    "gradient_checkpointing": True,
    "optim": "adamw_torch",
    "logging_steps": 10,
    "save_steps": 100,
    "eval_steps": 100,
    "save_total_limit": 3,
    "evaluation_strategy": "steps",
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "dataloader_num_workers": 4,
    "dataloader_pin_memory": True,
    "seed": 42,
}

PHASES = {
    1: {
        "name": "Foundation",
        "dataset_size": 20000,
        "learning_rate": 2e-4,
        "epochs": 3,
    },
    2: {
        "name": "Expansion",
        "dataset_size": 50000,
        "learning_rate": 1.5e-4,
        "epochs": 3,
    },
    3: {
        "name": "Optimization",
        "dataset_size": 80000,
        "learning_rate": 1e-4,
        "epochs": 3,
    },
    4: {
        "name": "Fine-tuning",
        "dataset_size": 10000,
        "learning_rate": 5e-5,
        "epochs": 2,
    },
}
