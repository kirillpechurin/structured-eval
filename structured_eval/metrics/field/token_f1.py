from __future__ import annotations

import re
from typing import Any

from structured_eval.metrics.protocol import FieldMetric

_NON_WORD = re.compile(r"[^\w\s]")


def _tokenize(value: Any) -> list[str]:
    """Lowercase, drop punctuation, split on whitespace."""
    return _NON_WORD.sub(" ", str(value).lower()).split()


class TokenF1(FieldMetric):
    """Continuous token-overlap F1 — a default for free-text fields."""

    name = "token_f1"

    def score(self, actual: Any, expected: Any) -> float:
        a = set(_tokenize(actual))
        e = set(_tokenize(expected))
        if not a and not e:
            return 1.0
        if not a or not e:
            return 0.0
        inter = len(a & e)
        precision = inter / len(a)
        recall = inter / len(e)
        denom = precision + recall
        return 2 * precision * recall / denom if denom else 0.0
