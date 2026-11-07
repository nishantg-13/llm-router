"""Benchmark questions: GSM8K (numeric) and a hard-subject MMLU slice (multiple choice)."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

# Harder MMLU subjects, chosen on purpose so small models do not already score near 100%.
MMLU_SUBJECTS = [
    "abstract_algebra",
    "college_mathematics",
    "college_physics",
    "formal_logic",
    "high_school_mathematics",
    "electrical_engineering",
]

GSM8K_SUFFIX = "\n\nSolve step by step. End your response with a final line of the form 'Answer: <number>'."
MMLU_SUFFIX = "\n\nAnswer with the letter only, in the form 'Answer: <letter>'."


@dataclass(frozen=True)
class Question:
    id: str
    source: str  # "gsm8k" or "mmlu"
    subject: str
    kind: str  # "num" or "mcq"
    prompt: str
    gold: str

    def to_dict(self) -> dict:
        return asdict(self)


def load_gsm8k(n: int, seed: int) -> list[Question]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = random.Random(seed).sample(range(len(ds)), n)
    return [
        Question(f"gsm8k-{i}", "gsm8k", "grade_school_math", "num", ds[i]["question"] + GSM8K_SUFFIX,
                 ds[i]["answer"].split("####")[-1].strip())
        for i in idx
    ]


def load_mmlu(n: int, seed: int) -> list[Question]:
    from datasets import load_dataset

    per = max(1, n // len(MMLU_SUBJECTS))
    out: list[Question] = []
    for subject in MMLU_SUBJECTS:
        ds = load_dataset("cais/mmlu", subject, split="test")
        for i in random.Random(f"{seed}-{subject}").sample(range(len(ds)), min(per, len(ds))):
            row = ds[i]
            options = "\n".join(f"{letter}. {text}" for letter, text in zip("ABCD", row["choices"], strict=True))
            out.append(Question(f"mmlu-{subject}-{i}", "mmlu", subject, "mcq",
                                f"{row['question']}\n\n{options}{MMLU_SUFFIX}", "ABCD"[row["answer"]]))
    return out[:n]


def load_questions(n_gsm8k: int, n_mmlu: int, seed: int) -> list[Question]:
    qs = load_gsm8k(n_gsm8k, seed) + load_mmlu(n_mmlu, seed)
    random.Random(seed).shuffle(qs)
    return qs
