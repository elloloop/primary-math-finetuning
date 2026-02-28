import json
from pathlib import Path

REQUIRED_FIELDS = {"question", "choices", "answer"}


def validate_record(record: dict) -> list[str]:
    errors = []
    missing = REQUIRED_FIELDS - set(record.keys())
    if missing:
        errors.append(f"Missing fields: {sorted(missing)}")
    choices = record.get("choices", [])
    if not isinstance(choices, list) or len(choices) != 4 or len(set(choices)) != 4:
        errors.append("choices must contain exactly 4 unique options")
    if record.get("answer") not in {"A", "B", "C", "D"}:
        errors.append("answer must be one of A,B,C,D")
    return errors


def validate_dataset(path: str) -> tuple[bool, list[str]]:
    data = json.loads(Path(path).read_text())
    errs = []
    seen = set()
    for i, item in enumerate(data):
        rec_errs = validate_record(item)
        q = item.get("question", "")
        if q in seen:
            rec_errs.append("duplicate question")
        seen.add(q)
        errs.extend([f"record {i}: {e}" for e in rec_errs])
    return (len(errs) == 0, errs)
