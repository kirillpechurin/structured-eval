"""The `numeric_closeness` metric — graded similarity rather than pass/fail."""

from __future__ import annotations

from typing import Any

from structured_eval.metrics.base import FieldMetric
from structured_eval.metrics.utils.null import both_null
from structured_eval.metrics.utils.number import parse_number


class NumericCloseness(FieldMetric):
    """Graded numeric similarity in `[0, 1]`, not a pass/fail tolerance.

    `1 - |actual - expected| / max(|actual|, |expected|)`: equal values score
    1.0, opposite signs trend toward 0.0, and `0/0` is 1.0. Being continuous
    rather than a 0/1 verdict, it is the default element scorer for numbers
    under the Hungarian aligner, where a graded cost matrix matters.

    Numbers only — a non-numeric side scores 0.0, `bool` included. Two `None`s
    are the exception and agree.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import NumericCloseness
        >>> from structured_eval.models import EvalConfig
        >>> NumericCloseness().score(95, 100)
        0.95
        >>> NumericCloseness().score(-5, 5)
        0.0
        >>> report = evaluate({"total": 95}, {"total": 100},
        ...                   EvalConfig(metrics=[NumericCloseness()]))
        >>> float(report.field_scores["total"].metrics["numeric_closeness"])
        0.95
    """

    name = "numeric_closeness"

    def score(self, actual: Any, expected: Any) -> float:
        """The smaller magnitude over the larger, in `[0, 1]`."""
        if both_null(actual, expected):
            return 1.0
        a = parse_number(actual)
        e = parse_number(expected)
        if a is None or e is None:
            return 0.0
        if a == e:
            return 1.0
        denom = max(abs(a), abs(e))
        return max(0.0, 1.0 - abs(a - e) / denom) if denom else 1.0
