"""Answer extraction and scoring. No LLM judge: quality is exact match on a number or a letter."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _norm_number(raw: str) -> str | None:
    cleaned = raw.replace(",", "").replace("$", "").strip()
    try:
        d = Decimal(cleaned)
    except InvalidOperation:
        return None
    text = format(d.normalize(), "f")
    return "0" if text in ("-0", "") else text


_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_ANSWER_LINE = re.compile(r"(?i:answer)\s*[:=]\s*(.*)")
_L = r"(?:\(([A-Da-d])\)|([A-D])\b)"
_LETTER_PATTERNS = [
    re.compile(r"(?i:answer)\s*[:=]\s*[*\s$\\{]*(?:boxed\{)?[*\s]*" + _L),
    re.compile(r"\\boxed\{\s*(?:\(([A-Da-d])\)|([A-D]))\s*\}"),
    re.compile(r"(?i:(?:final answer|correct (?:answer|option|choice)) is)\s*[:*\s$\\]*" + _L),
]


def extract_number(text: str) -> str | None:
    """Final number: the last 'Answer: <n>' or \\boxed{<n>} statement, else the last number in the text."""
    cands = [(m.start(), m.group(1)) for m in _ANSWER_LINE.finditer(text)]
    cands += [(m.start(), m.group(1)) for m in _BOXED.finditer(text)]
    for _, chunk in sorted(cands, reverse=True):
        found = _NUM.search(chunk.replace("$", ""))
        if found:
            return _norm_number(found.group())
    nums = _NUM.findall(text)
    return _norm_number(nums[-1]) if nums else None


def extract_letter(text: str) -> str | None:
    """Final multiple-choice letter: the last of 'Answer: X', \\boxed{X}, 'final answer is X', 'correct option is X'."""
    best: tuple[int, str] | None = None
    for pat in _LETTER_PATTERNS:
        for m in pat.finditer(text):
            letter = (m.group(1) or m.group(2)).upper()
            if best is None or m.start() > best[0]:
                best = (m.start(), letter)
    if best:
        return best[1]
    m = re.fullmatch(r"\s*\(?([A-Da-d])\)?[.)]?\s*", text)
    return m.group(1).upper() if m else None


def is_correct(kind: str, response: str, gold: str) -> bool:
    if kind == "num":
        pred, want = extract_number(response), _norm_number(gold)
        return pred is not None and pred == want
    if kind == "mcq":
        pred = extract_letter(response)
        return pred is not None and pred == gold.upper()
    raise ValueError(f"unknown question kind {kind!r}")
