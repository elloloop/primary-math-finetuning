import json
from collections import Counter


def summarize_errors(records: list[dict], out_path: str):
    by_type = Counter(r.get("error_type", "other") for r in records)
    payload = {
        "total_errors": len(records),
        "error_types": dict(by_type),
        "sample_errors": records[:10],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
