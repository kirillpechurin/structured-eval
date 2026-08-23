"""The `fuzzy` metric — string similarity via RapidFuzz (the `fuzzy` extra)."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from structured_eval.metrics.base import FieldMetric
from structured_eval.metrics.utils.null import both_null

_IGNORE_WHITESPACE_REGEX = re.compile(r"\s+")


class FuzzyMethod(StrEnum):
    """RapidFuzz scorer used by `Fuzzy`.

    Attributes:
        RATIO: Plain normalized Levenshtein ratio.
        PARTIAL_RATIO: Best matching substring.
        TOKEN_SORT_RATIO: Order-insensitive; sorts tokens first.
        TOKEN_SET_RATIO: Set-based; ignores duplicate and extra tokens.
    """

    RATIO = "ratio"
    PARTIAL_RATIO = "partial_ratio"
    TOKEN_SORT_RATIO = "token_sort_ratio"
    TOKEN_SET_RATIO = "token_set_ratio"


class Fuzzy(FieldMetric):
    """Fuzzy string similarity via RapidFuzz, behind the `fuzzy` extra.

    String-only: a non-`str` side scores 0.0, with no coercion. Two `None`s are
    the exception and agree.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import Fuzzy
        >>> from structured_eval.metrics.fuzzy import FuzzyMethod
        >>> from structured_eval.models import EvalConfig
        >>> Fuzzy().score("Acme  Corp", "acme corp")
        1.0
        >>> round(Fuzzy().score("Acme Corporation", "Acme Corp"), 3)
        0.72
        >>> partial = Fuzzy(method=FuzzyMethod.PARTIAL_RATIO)
        >>> partial.score("Acme Corporation", "Acme Corp")   # best substring
        1.0
        >>> report = evaluate({"vendor": "acme  corp"}, {"vendor": "Acme Corp"},
        ...                   EvalConfig(metrics=[Fuzzy()]))
        >>> float(report.field_scores["vendor"].metrics["fuzzy"])
        1.0
    """

    name = "fuzzy"

    def __init__(
        self,
        method: FuzzyMethod = FuzzyMethod.TOKEN_SORT_RATIO,
        ignore_case: bool = True,
        ignore_whitespace: bool = True,
        name: str | None = None,
    ):
        """Pick the scorer and the normalization applied before it.

        Args:
            method: Which RapidFuzz scorer to use.
            ignore_case: Lowercase both sides first.
            ignore_whitespace: Collapse runs of whitespace and trim the ends.
            name: Per-instance report key.

        The two normalizations are independent, so a case-insensitive but
        whitespace-sensitive comparison is expressible, or the reverse.
        """
        super().__init__(name=name)
        self.method = FuzzyMethod(method)
        self.ignore_case = ignore_case
        self.ignore_whitespace = ignore_whitespace

    def score(self, actual: Any, expected: Any) -> float:
        """RapidFuzz's similarity for the two strings, rescaled to `[0, 1]`."""
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
            a = _IGNORE_WHITESPACE_REGEX.sub(" ", a).strip()
            e = _IGNORE_WHITESPACE_REGEX.sub(" ", e).strip()
        if self.ignore_case:
            a, e = a.lower(), e.lower()
        return float(scorer(a, e)) / 100.0
