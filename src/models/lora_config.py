from peft import LoraConfig
from config.model_config import LORA_CONFIG


def build_lora_config(r: int | None = None) -> LoraConfig:
    cfg = dict(LORA_CONFIG)
    if r is not None:
        cfg["r"] = r
        cfg["lora_alpha"] = 2 * r
    return LoraConfig(**cfg)
