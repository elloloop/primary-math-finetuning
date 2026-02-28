from src.models.lora_config import build_lora_config


def test_lora_config():
    cfg = build_lora_config(32)
    assert cfg.r == 32
    assert cfg.lora_alpha == 64
