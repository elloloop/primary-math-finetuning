import json
from pathlib import Path
from src.data.data_formatter import to_chatml
from src.data.data_validator import validate_record, validate_dataset


def test_format_qwen_chat():
    record = {"question": "1+1?", "choices": ["1", "2", "3", "4"], "answer": "B", "explanation": "1+1=2"}
    txt = to_chatml(record)
    assert "<|im_start|>system" in txt
    assert "The correct answer is: B" in txt


def test_data_validation(tmp_path: Path):
    rec = {"question": "1+1?", "choices": ["1", "2", "3", "4"], "answer": "B"}
    assert validate_record(rec) == []
    p = tmp_path / "d.json"
    p.write_text(json.dumps([rec]))
    ok, errs = validate_dataset(str(p))
    assert ok
    assert not errs
