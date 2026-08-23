"""The `exponential_numeric_score` metric — similarity decaying with distance."""

from __future__ import annotations

import math
from typing import Any

from structured_eval.metrics.base import FieldMetric
from structured_eval.metrics.utils.null import both_null
from structured_eval.metrics.utils.number import parse_number


class ExponentialNumericScore(FieldMetric):
    """Exponentially decaying similarity for numeric fields.

    `exp(-abs(actual - expected) / scale)`: 1.0 for an exact match, decaying
    smoothly to values in `(0.0, 1.0]`. Unlike the ratio-based
    `NumericCloseness`, the decay is on the **absolute** error, which makes it
    unit-aware.

    Numbers only — a non-numeric side scores 0.0, `bool` included. Two `None`s
    are the exception and agree.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import ExponentialNumericScore
        >>> from structured_eval.models import EvalConfig
        >>> ExponentialNumericScore(scale=10).score(100, 100)
        1.0
        >>> round(ExponentialNumericScore(scale=10).score(105, 100), 3)
        0.607
        >>> round(ExponentialNumericScore(scale=100).score(105, 100), 3)  # tolerant
        0.951
        >>> report = evaluate({"total": 105}, {"total": 100},
        ...                   EvalConfig(metrics=[ExponentialNumericScore(scale=10)]))
        >>> key = "exponential_numeric_score"
        >>> round(float(report.field_scores["total"].metrics[key]), 3)
        0.607
    """

    name = "exponential_numeric_score"

    def __init__(self, scale: float = 1.0, name: str | None = None) -> None:
        """Set how fast the score decays with distance.

        Args:
            scale: The error at which the score falls to `1/e`; pick it to match
                the field's units. Larger is more tolerant.
            name: Per-instance report key.

        Raises:
            ValueError: If `scale` is not greater than 0.
        """
        super().__init__(name=name)
        if scale <= 0:
            raise ValueError("scale must be greater than 0")
        self.scale = scale

    def score(self, actual: Any, expected: Any) -> float:
        """Similarity decaying exponentially with the absolute error."""
        if both_null(actual, expected):
            return 1.0
        a = parse_number(actual)
        e = parse_number(expected)
        if a is None or e is None:
            return 0.0
        return math.exp(-abs(a - e) / self.scale)
