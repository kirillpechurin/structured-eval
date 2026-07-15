from __future__ import annotations

from enum import StrEnum
from typing import Any

from structured_eval.metrics.base import FieldMetric
from structured_eval.metrics.utils.null import both_null


class FuzzyMethod(StrEnum):
    """RapidFuzz scorer used by :class:`Fuzzy`."""

    RATIO = "ratio"  # plain normalized Levenshtein ratio
    PARTIAL_RATIO = "partial_ratio"  # best matching substring
    TOKEN_SORT_RATIO = "token_sort_ratio"  # order-insensitive (default)
    TOKEN_SET_RATIO = "token_set_ratio"  # set-based, ignores duplicate tokens


class Fuzzy(FieldMetric):
    """Fuzzy string similarity via RapidFuzz (optional dependency).

    ``method`` selects the RapidFuzz scorer:

    * ``ratio`` — plain normalized Levenshtein ratio;
    * ``partial_ratio`` — best matching substring;
    * ``token_sort_ratio`` (default) — order-insensitive, sorts tokens;
    * ``token_set_ratio`` — set-based, ignores duplicate/extra tokens.

    ``ignore_case`` lowercases and ``ignore_whitespace`` strips surrounding
    whitespace before comparison; the two are independent, so a
    case-insensitive but whitespace-sensitive comparison (or the reverse) is
    expressible. Both default to ``True``.
    String-only: if either side is not a ``str`` the score is 0.0 (no coercion),
    consistent with the other text metrics — except two ``None``s, which agree
    (1.0; see ``metrics.utils.null``).
    """

    name = "fuzzy"

    def __init__(
        self,
        method: FuzzyMethod = FuzzyMethod.TOKEN_SORT_RATIO,
        ignore_case: bool = True,
        ignore_whitespace: bool = True,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.method = FuzzyMethod(method)
        self.ignore_case = ignore_case
        self.ignore_whitespace = ignore_whitespace

    def score(self, actual: Any, expected: Any) -> float:
        if both_null(actual, expected):
            return 1.0
        if not (isinstance(actual, str) and isinstance(expected, str)):
            return 0.0
        try:
            from rapidfuzz import fuzz
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "rapidfuzz is required for the 'fuzzy' metric. "
                "Install it with: pip install 'structured-eval[fuzzy]'"
            ) from exc

        scorer = {
            "ratio": fuzz.ratio,
            "partial_ratio": fuzz.partial_ratio,
            "token_sort_ratio": fuzz.token_sort_ratio,
            "token_set_ratio": fuzz.token_set_ratio,
        }[self.method]

        a, e = actual, expected
        if self.ignore_whitespace:
            a, e = a.strip(), e.strip()
        if self.ignore_case:
            a, e = a.lower(), e.lower()
        return float(scorer(a, e)) / 100.0
