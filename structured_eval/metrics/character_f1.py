from __future__ import annotations

import re
from collections import Counter
from typing import Any

from structured_eval.metrics.base import FieldMetric

_NON_WORD = re.compile(r"[^\w\s]")


def _characters(value: Any) -> list[str]:
    """Lowercase, drop punctuation and whitespace, split into characters."""
    normalized = _NON_WORD.sub("", str(value).lower())
    normalized = "".join(normalized.split())  # remove all whitespace
    return list(normalized)


class CharacterF1(FieldMetric):
    """Character-overlap F1 for short free-text fields.

    Characters are matched as a **multiset** (``Counter``), so repeated
    characters contribute only as many times as they appear on both sides.
    Precision and recall are computed over character counts, and their
    harmonic mean is returned. String-only: if either side is not a ``str``
    the score is ``0.0`` (no coercion).
    """

    name = "character_f1"

    def score(self, actual: Any, expected: Any) -> float:
        if not (isinstance(actual, str) and isinstance(expected, str)):
            return 0.0

        a = _characters(actual)
        e = _characters(expected)

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
