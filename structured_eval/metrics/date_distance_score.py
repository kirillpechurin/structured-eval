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

    The score is computed as::

        max(0, 1 - days_difference / max_days)

    yielding:

    - ``1.0`` for identical dates;
    - a linear decrease as the difference in days grows;
    - ``0.0`` once the difference reaches or exceeds ``max_days``.

    Both ``date`` and ``datetime`` values are supported, and ISO-8601 strings
    (e.g. ``"2026-06-29"``) are coerced via pydantic. Datetime values are
    compared by their calendar date only (time-of-day is ignored).

    If either side cannot be read as a date — ``None``, an unparseable string,
    or any non-date type — the score is ``0.0``. Two ``None``s are the exception
    — no date was expected and none was given, so they agree (``1.0``; see
    ``metrics.utils.null``).
    """

    name = "date_distance_score"

    def __init__(self, max_days: int = 30, name: str | None = None) -> None:
        super().__init__(name=name)
        if max_days <= 0:
            raise ValueError("max_days must be greater than 0")
        self.max_days = max_days

    def score(self, actual: Any, expected: Any) -> float:
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
