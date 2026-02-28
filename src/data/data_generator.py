import json
import random


def generate_samples(num_samples: int) -> list[dict]:
    samples = []
    for _ in range(num_samples):
        a, b = random.randint(1, 200), random.randint(1, 200)
        answer = a + b
        wrong = [answer + 1, answer - 1, answer + 10]
        choices = [answer] + wrong
        random.shuffle(choices)
        idx = choices.index(answer)
        samples.append(
            {
                "question": f"What is {a} + {b}?",
                "choices": [str(c) for c in choices],
                "answer": chr(65 + idx),
                "explanation": f"{a} + {b} = {answer}",
                "difficulty": "easy",
                "category": "addition",
            }
        )
    return samples


def save_samples(samples: list[dict], output: str) -> None:
    with open(output, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
