import json
from pathlib import Path
from datasets import Dataset
from src.data.data_formatter import to_chatml
from src.data.data_validator import validate_dataset


def load_training_dataset(path: str, quick_test: bool = False) -> Dataset:
    ok, errs = validate_dataset(path)
    if not ok:
        raise ValueError("Dataset validation failed:\n" + "\n".join(errs[:20]))
    records = json.loads(Path(path).read_text())
    if quick_test:
        records = records[:100]
    texts = [{"text": to_chatml(r)} for r in records]
    return Dataset.from_list(texts)
