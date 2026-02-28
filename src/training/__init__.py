"""Training package -- model fine-tuning orchestration, callbacks, and utilities."""

from src.training.callbacks import (
    EarlyStoppingWithPatience,
    GPUMemoryCallback,
    TrainingProgressCallback,
)
from src.training.trainer import MathTrainer
from src.training.utils import (
    build_data_collator,
    check_gpu_availability,
    compute_metrics,
    estimate_training_time,
    get_training_args,
)

__all__ = [
    "MathTrainer",
    "GPUMemoryCallback",
    "EarlyStoppingWithPatience",
    "TrainingProgressCallback",
    "build_data_collator",
    "compute_metrics",
    "get_training_args",
    "check_gpu_availability",
    "estimate_training_time",
]
