from __future__ import annotations

import re
from collections import Counter
from typing import Any

from structured_eval.metrics.base import FieldMetric

_IGNORE_PUNCTUATION_REGEX = re.compile(r"[^\w\s]")
_IGNORE_WHITESPACE_REGEX = re.compile(r"\s+")


class CharacterF1(FieldMetric):
    """Character-overlap F1 for short free-text fields.

    Characters are matched as a **multiset** (``Counter``), so repeated
    characters contribute only as many times as they appear on both sides.
    Precision and recall are computed over character counts, and their
    harmonic mean is returned. String-only: if either side is not a ``str``
    the score is ``0.0`` (no coercion).

    Normalization is applied to both sides before the comparison and each
    step can be turned off independently::

        CharacterF1(ignore_case=False)         # "AB" vs "ab" scores below 1.0
        CharacterF1(ignore_punctuation=False)  # "," and "." count as characters
        CharacterF1(ignore_whitespace=False)   # spaces count as characters

    The defaults keep every normalization on.
    """

    name = "character_f1"

    def __init__(
        self,
        ignore_case: bool = True,
        ignore_whitespace: bool = True,
        ignore_punctuation: bool = True,
    ):
        self.ignore_case = ignore_case
        self.ignore_whitespace = ignore_whitespace
        self.ignore_punctuation = ignore_punctuation

    def _characters(self, value: str) -> list[str]:
        if self.ignore_case:
            value = value.lower()
        if self.ignore_punctuation:
            value = _IGNORE_PUNCTUATION_REGEX.sub("", value)
        if self.ignore_whitespace:
            value = _IGNORE_WHITESPACE_REGEX.sub("", value)
        return list(value)

    def score(self, actual: Any, expected: Any) -> float:
        if not (isinstance(actual, str) and isinstance(expected, str)):
            return 0.0

        a = self._characters(actual)
        e = self._characters(expected)

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
