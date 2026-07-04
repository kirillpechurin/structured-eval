"""DateDistanceScore — linear ``max(0, 1 - days/max_days)`` date similarity.

Accepts ``date`` / ``datetime`` and ISO-8601 strings (coerced via pydantic);
datetime is compared by calendar date only. Any unparseable side scores 0.0.
``max_days`` must be > 0.
"""

from datetime import date, datetime
from typing import Any

import pytest

from structured_eval.metrics import DateDistanceScore

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("actual", "expected", "max_days", "score"),
    [
        (date(2026, 6, 29), date(2026, 6, 29), 30, 1.0),  # identical
        (date(2026, 6, 29), date(2026, 7, 4), 30, 1 - 5 / 30),  # 5 days apart
        (date(2026, 6, 29), date(2026, 8, 29), 30, 0.0),  # ≥ max_days → 0.0
        ("2026-06-29", "2026-07-04", 30, 1 - 5 / 30),  # ISO strings coerced
        (datetime(2026, 6, 29, 23, 59), date(2026, 6, 29), 30, 1.0),  # time ignored
        (date(2026, 7, 4), date(2026, 6, 29), 30, 1 - 5 / 30),  # symmetric
    ],
    ids=["identical", "5-days", "beyond-max", "iso-strings", "datetime", "symmetric"],
)
def test_date_distance(actual: Any, expected: Any, max_days: int, score: Any) -> None:
    assert DateDistanceScore(max_days=max_days).score(
        actual, expected
    ) == pytest.approx(score)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ("not-a-date", date(2026, 6, 29)),  # unparseable string
        (None, date(2026, 6, 29)),  # null
        (42, date(2026, 6, 29)),  # non-date type
    ],
    ids=["unparseable", "null", "non-date"],
)
def test_non_date_is_zero(actual: Any, expected: Any) -> None:
    assert DateDistanceScore().score(actual, expected) == 0.0


@pytest.mark.parametrize("max_days", [0, -10], ids=["zero", "negative"])
def test_non_positive_max_days_rejected(max_days: int) -> None:
    with pytest.raises(ValueError, match="max_days must be greater than 0"):
        DateDistanceScore(max_days=max_days)
