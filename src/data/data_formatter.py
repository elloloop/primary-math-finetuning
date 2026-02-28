SYSTEM_MESSAGE = "You are a helpful math tutor. Solve problems step by step and provide the correct answer."


def to_chatml(record: dict) -> str:
    opts = "\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(record["choices"]))
    explanation = record.get("explanation", "")
    return (
        f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n"
        f"<|im_start|>user\n{record['question']}\n\nOptions:\n{opts}<|im_end|>\n"
        f"<|im_start|>assistant\n{explanation}\nThe correct answer is: {record['answer']}<|im_end|>"
    )
