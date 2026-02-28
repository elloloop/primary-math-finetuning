from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model
from config.model_config import MODEL_CONFIG
from src.models.lora_config import build_lora_config


def load_model_and_tokenizer(model_name: str | None = None, lora_r: int | None = None):
    name = model_name or MODEL_CONFIG["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model = get_peft_model(model, build_lora_config(lora_r))
    return model, tokenizer
