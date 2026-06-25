from __future__ import annotations

import re
from collections import Counter
from typing import Any

from structured_eval.metrics.base import FieldMetric

_NON_WORD = re.compile(r"[^\w\s]")


def _tokenize(value: Any) -> list[str]:
    """Lowercase, drop punctuation, split on whitespace."""
    return _NON_WORD.sub(" ", str(value).lower()).split()


class TokenF1(FieldMetric):
    """SQuAD-style token-overlap F1 — a default for free-text fields.

    Tokens are matched as a **multiset** (``Counter``), counting shared tokens
    with multiplicity exactly like the official SQuAD F1 — so a repeated token
    only helps as often as it appears on both sides (``"the the cat"`` vs
    ``"the cat"`` is 0.8, not 1.0). Precision and recall are over the token
    *counts*; their harmonic mean is the score.
    """

    name = "token_f1"

    def score(self, actual: Any, expected: Any) -> float:
        a = _tokenize(actual)
        e = _tokenize(expected)
        if not a and not e:
            return 1.0
        if not a or not e:
            return 0.0
        same = sum((Counter(a) & Counter(e)).values())
        if not same:
            return 0.0
        precision = same / len(a)
        recall = same / len(e)
        return 2 * precision * recall / (precision + recall)
