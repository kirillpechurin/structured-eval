"""The `token_f1` metric — SQuAD-style token-overlap F1 for free text."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any

from structured_eval.metrics.base import FieldMetric
from structured_eval.metrics.utils.null import both_null

_IGNORE_PUNCTUATION_CHARS = frozenset(string.punctuation)
_IGNORE_ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


class TokenF1(FieldMetric):
    """SQuAD-style token-overlap F1 — a default for free-text fields.

    On its defaults this reproduces the `f1_score` of the official SQuAD v1.1
    script: both sides are lowercased, stripped of punctuation and of the
    articles a/an/the, and their whitespace collapsed. Tokens then match as a
    multiset, so a repeated token helps only as often as both sides carry it.

    Two deliberate departures, because this scores fields rather than answers:
    two empty strings score 1.0 where the script returns 0.0, and a non-`str`
    side scores 0.0 with no coercion. Two `None`s agree.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import TokenF1
        >>> from structured_eval.models import EvalConfig
        >>> TokenF1().score("the quick brown fox", "a quick brown fox")
        1.0
        >>> round(TokenF1().score("quick brown fox", "the brown fox jumps"), 3)
        0.667
        >>> TokenF1(ignore_articles=False).score("the the cat", "the cat")
        0.8
        >>> report = evaluate({"summary": "quick brown fox"},
        ...                   {"summary": "the brown fox jumps"},
        ...                   EvalConfig(metrics=[TokenF1()]))
        >>> round(float(report.field_scores["summary"].metrics["token_f1"]), 3)
        0.667
    """

    name = "token_f1"

    def __init__(
        self,
        ignore_case: bool = True,
        ignore_punctuation: bool = True,
        ignore_articles: bool = True,
        name: str | None = None,
    ):
        """Choose which normalizations run before tokenizing.

        Args:
            ignore_case: Lowercase both sides.
            ignore_punctuation: Drop the ASCII punctuation of
                `string.punctuation`.
            ignore_articles: Drop `a` / `an` / `the`, as the SQuAD script does.
            name: Per-instance report key.
        """
        super().__init__(name=name)
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.ignore_articles = ignore_articles

    def _tokenize(self, value: str) -> list[str]:
        """The string as the tokens this instance compares, in order."""
        if self.ignore_case:
            value = value.lower()
        if self.ignore_punctuation:
            value = "".join(ch for ch in value if ch not in _IGNORE_PUNCTUATION_CHARS)
        if self.ignore_articles:
            value = _IGNORE_ARTICLES_REGEX.sub(" ", value)
        return value.split()

    def score(self, actual: Any, expected: Any) -> float:
        """Harmonic mean of token precision and recall over the two strings."""
        if both_null(actual, expected):
            return 1.0
        if not (isinstance(actual, str) and isinstance(expected, str)):
            return 0.0

        a = self._tokenize(actual)
        e = self._tokenize(expected)

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
