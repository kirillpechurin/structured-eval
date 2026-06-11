from __future__ import annotations

from typing import Any

from structured_eval.matchers.protocol import MatcherBase
from structured_eval.matchers.token_f1 import _tokenize


class JaccardMatcher(MatcherBase):
    """Token-level Jaccard similarity: ``|A ∩ B| / |A ∪ B|``."""

    name = "JACCARD"

    def similarity(self, actual: Any, expected: Any) -> float:
        a = set(_tokenize(actual))
        e = set(_tokenize(expected))
        if not a and not e:
            return 1.0
        if not a or not e:
            return 0.0
        return len(a & e) / len(a | e)
