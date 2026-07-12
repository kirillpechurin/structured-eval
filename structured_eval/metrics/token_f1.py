from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any

from structured_eval.metrics.base import FieldMetric

_IGNORE_PUNCTUATION_CHARS = frozenset(string.punctuation)
_IGNORE_ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


class TokenF1(FieldMetric):
    """SQuAD-style token-overlap F1 — a default for free-text fields.

    On its defaults this reproduces the ``f1_score`` of the official SQuAD v1.1
    evaluation script: both sides go through the reference ``normalize_answer``
    (lowercase, drop punctuation, drop the articles ``a``/``an``/``the``, collapse
    whitespace), then tokens are matched as a **multiset** (``Counter``) — a
    repeated token only helps as often as it appears on both sides, so
    ``"the the cat"`` vs ``"the cat"`` is 0.8, not 1.0. Precision and recall are
    over the token *counts*; their harmonic mean is the score.

    Each normalization step can be turned off independently::

        TokenF1(ignore_case=False)         # "AB" vs "ab" scores below 1.0
        TokenF1(ignore_punctuation=False)  # "fox." and "fox" are distinct tokens
        TokenF1(ignore_articles=False)     # "the" counts as a token like any other

    Two deliberate departures from the reference script, both because this scores
    fields rather than question answers: two empty strings score 1.0 (the script
    returns 0.0, an empty answer being a failed answer), and a value that is not a
    ``str`` scores 0.0 with no coercion.
    """

    name = "token_f1"

    def __init__(
        self,
        ignore_case: bool = True,
        ignore_punctuation: bool = True,
        ignore_articles: bool = True,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.ignore_articles = ignore_articles

    def _tokenize(self, value: str) -> list[str]:
        if self.ignore_case:
            value = value.lower()
        if self.ignore_punctuation:
            value = "".join(ch for ch in value if ch not in _IGNORE_PUNCTUATION_CHARS)
        if self.ignore_articles:
            value = _IGNORE_ARTICLES_REGEX.sub(" ", value)
        return value.split()

    def score(self, actual: Any, expected: Any) -> float:
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
