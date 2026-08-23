"""The `date_distance_score` metric — graded similarity for dates and datetimes."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import TypeAdapter

from structured_eval.metrics.base import FieldMetric
from structured_eval.metrics.utils.null import both_null


def _to_date(value: Any) -> date | None:
    try:
        adapter = TypeAdapter(date)
        return adapter.validate_python(value)
    except Exception:
        return None


class DateDistanceScore(FieldMetric):
    """Linear similarity for date and datetime fields.

    `max(0, 1 - days_difference / max_days)`: 1.0 for identical dates, falling
    linearly to 0.0 once the gap reaches `max_days`. `date`, `datetime` and
    ISO-8601 strings are all accepted, compared by calendar date only.

    A side that cannot be read as a date scores 0.0. Two `None`s are the
    exception and agree.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import DateDistanceScore
        >>> from structured_eval.models import EvalConfig
        >>> DateDistanceScore().score("2024-01-01", "2024-01-01")
        1.0
        >>> round(DateDistanceScore(max_days=30).score("2024-01-08", "2024-01-01"), 3)
        0.767
        >>> DateDistanceScore(max_days=3).score("2024-01-08", "2024-01-01")
        0.0
        >>> report = evaluate({"issued": "2024-01-08"}, {"issued": "2024-01-01"},
        ...                   EvalConfig(metrics=[DateDistanceScore(max_days=30)]))
        >>> round(float(report.field_scores["issued"].metrics["date_distance_score"]), 3)
        0.767
    """

    name = "date_distance_score"

    def __init__(self, max_days: int = 30, name: str | None = None) -> None:
        """Set how far apart two dates may be before scoring 0.

        Args:
            max_days: The gap at which the score reaches 0.0.
            name: Per-instance report key.

        Raises:
            ValueError: If `max_days` is not greater than 0.
        """
        super().__init__(name=name)
        if max_days <= 0:
            raise ValueError("max_days must be greater than 0")
        self.max_days = max_days

    def score(self, actual: Any, expected: Any) -> float:
        """Similarity falling linearly with the gap in calendar days."""
        if both_null(actual, expected):
            return 1.0
        if not isinstance(actual, (date, datetime)):
            actual = _to_date(actual)
        if not isinstance(expected, (date, datetime)):
            expected = _to_date(expected)
        if not (
            isinstance(actual, (date, datetime))
            and isinstance(expected, (date, datetime))
        ):
            return 0.0

        actual_date = actual.date() if isinstance(actual, datetime) else actual
        expected_date = expected.date() if isinstance(expected, datetime) else expected

        days = abs((actual_date - expected_date).days)

        return max(0.0, 1.0 - days / self.max_days)
