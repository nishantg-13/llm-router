"""Routing policies. A policy maps a prompt to a tier name: "small" or "large"."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Protocol

import numpy as np


class Policy(Protocol):
    name: str

    def choose(self, prompt: str) -> str: ...


class AlwaysSmall:
    name = "always_small"

    def choose(self, prompt: str) -> str:
        return "small"


class AlwaysLarge:
    name = "always_large"

    def choose(self, prompt: str) -> str:
        return "large"


# Fixed before any held-out result was looked at. The rules router is a deliberately simple baseline:
# prompts with LaTeX commands, math symbols or heavy technical vocabulary, or very long prompts, go to the large model.
_CUES = [
    re.compile(r"\\[a-zA-Z]{2,}"),
    re.compile(r"[\u222b\u2211\u221a\u2264\u2265\u2260\u2208]"),
    re.compile(
        r"\b(?:prove|derivative|integral|eigen\w*|polynomial|subgroup|isomorphi\w*|differential equation)\b",
        re.IGNORECASE,
    ),
]


class RulesRouter:
    name = "rules"

    def __init__(self, max_chars: int = 700):
        self.max_chars = max_chars

    def choose(self, prompt: str) -> str:
        if len(prompt) > self.max_chars or any(c.search(prompt) for c in _CUES):
            return "large"
        return "small"


class LearnedRouter:
    """Predicts whether the small model will answer correctly from a prompt embedding; routes to the small
    model when that probability is at least `threshold`, otherwise to the large model."""

    name = "learned"

    def __init__(self, embed: Callable[[list[str]], np.ndarray], threshold: float = 0.9, C: float = 1.0):
        self.embed, self.threshold, self.C = embed, threshold, C
        self.clf = None
        self._const: float | None = None

    def fit(self, prompts: list[str], small_correct: list[bool]) -> LearnedRouter:
        from sklearn.linear_model import LogisticRegression

        y = np.asarray(small_correct, dtype=int)
        if y.min() == y.max():  # one class only: nothing to learn, predict the constant
            self._const, self.clf = float(y[0]), None
        else:
            self.clf = LogisticRegression(C=self.C, max_iter=2000).fit(self.embed(prompts), y)
        return self

    def p_small_ok(self, prompts: list[str]) -> np.ndarray:
        if self.clf is None:
            if self._const is None:
                raise RuntimeError("LearnedRouter is not fitted")
            return np.full(len(prompts), self._const)
        return self.clf.predict_proba(self.embed(prompts))[:, 1]

    def choose(self, prompt: str) -> str:
        return "small" if self.p_small_ok([prompt])[0] >= self.threshold else "large"


def default_policies() -> dict[str, Policy]:
    """Policies available to the live gateway without a trained model."""
    return {p.name: p for p in (AlwaysSmall(), AlwaysLarge(), RulesRouter())}
