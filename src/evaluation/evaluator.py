import re


class BaseEvaluator:
    PATTERNS = [
        r"####\s*(-?\d+\.?\d*)",
        r"answer is\s*([A-Da-d])",
        r"correct answer is\s*([A-Da-d])",
        r"^([A-Da-d])\)",
        r"(-?\d+\.?\d*)$",
    ]

    def extract_answer(self, text: str) -> str | None:
        for p in self.PATTERNS:
            m = re.search(p, text, re.IGNORECASE | re.MULTILINE)
            if m:
                return m.group(1).strip().upper()
        return None
