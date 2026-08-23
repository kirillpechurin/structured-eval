"""The `character_f1` metric — character-overlap F1 for short free text."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any

from structured_eval.metrics.base import FieldMetric
from structured_eval.metrics.utils.null import both_null

_IGNORE_PUNCTUATION_CHARS = frozenset(string.punctuation)
_IGNORE_WHITESPACE_REGEX = re.compile(r"\s+")


class CharacterF1(FieldMetric):
    """Character-overlap F1 for short free-text fields.

    Characters are matched as a multiset, so a repeated character helps only as
    often as it appears on both sides. Precision and recall run over the counts,
    and the score is their harmonic mean. String-only: a non-`str` side scores
    0.0, and two `None`s agree.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import CharacterF1
        >>> from structured_eval.models import EvalConfig
        >>> round(CharacterF1().score("color", "colour"), 3)
        0.909
        >>> CharacterF1().score("Acme, Inc.", "acme inc")
        1.0
        >>> report = evaluate({"vendor": "Acme, Inc."}, {"vendor": "acme inc"},
        ...                   EvalConfig(metrics=[CharacterF1()]))
        >>> float(report.field_scores["vendor"].metrics["character_f1"])
        1.0
    """

    name = "character_f1"

    def __init__(
        self,
        ignore_case: bool = True,
        ignore_whitespace: bool = True,
        ignore_punctuation: bool = True,
        name: str | None = None,
    ):
        """Choose which normalizations run before the comparison.

        Args:
            ignore_case: Lowercase both sides.
            ignore_whitespace: Drop whitespace entirely.
            ignore_punctuation: Drop the ASCII punctuation of
                `string.punctuation` — the same set `TokenF1` uses, so `_` goes
                and non-ASCII punctuation such as `«»—` stays.
            name: Per-instance report key.
        """
        super().__init__(name=name)
        self.ignore_case = ignore_case
        self.ignore_whitespace = ignore_whitespace
        self.ignore_punctuation = ignore_punctuation

    def _characters(self, value: str) -> list[str]:
        if self.ignore_case:
            value = value.lower()
        if self.ignore_punctuation:
            value = "".join(ch for ch in value if ch not in _IGNORE_PUNCTUATION_CHARS)
        if self.ignore_whitespace:
            value = _IGNORE_WHITESPACE_REGEX.sub("", value)
        return list(value)

    def score(self, actual: Any, expected: Any) -> float:
        """Harmonic mean of character precision and recall over the two strings."""
        if both_null(actual, expected):
            return 1.0
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
