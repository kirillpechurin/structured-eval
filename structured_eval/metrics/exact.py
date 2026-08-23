"""The `exact_match` metric — strict `actual == expected`."""

from __future__ import annotations

from typing import Any

from structured_eval.metrics.base import FieldMetric


class ExactMatch(FieldMetric):
    """Strict equality: `actual == expected` → 1.0, else 0.0.

    The default scalar comparison, and the default key comparison in `by_key`
    array alignment. It does not score whole objects or arrays — those read
    their children's representatives instead.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import ExactMatch
        >>> from structured_eval.models import EvalConfig
        >>> ExactMatch().score("paid", "paid")
        1.0
        >>> ExactMatch().score("100", 100)      # type-sensitive
        0.0
        >>> report = evaluate({"status": "paid", "code": "100"},
        ...                   {"status": "paid", "code": 100},
        ...                   EvalConfig(metrics=[ExactMatch()]))
        >>> float(report.field_scores["code"].metrics["exact_match"])
        0.0
    """

    name = "exact_match"

    def score(self, actual: Any, expected: Any) -> float:
        """1.0 when the two values are equal, else 0.0."""
        return 1.0 if actual == expected else 0.0
