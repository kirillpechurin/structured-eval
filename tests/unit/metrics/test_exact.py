"""ExactMatch — strict, type-sensitive equality.

A field metric *is* the comparison: ``score(actual, expected) -> float`` is a
pure primitive, tested without the engine.
"""

from typing import Any

import pytest

from structured_eval import ExactMatch

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("actual", "expected", "score"),
    [
        ("paid", "paid", 1.0),
        ("paid", "draft", 0.0),
        ("100", 100, 0.0),  # type-sensitive: str != int
        (100, 100, 1.0),
        (None, None, 1.0),  # null == null
    ],
    ids=["equal", "differ", "type-sensitive", "equal-int", "null-eq-null"],
)
def test_score(actual: Any, expected: Any, score: Any) -> None:
    assert ExactMatch().score(actual, expected) == score


def test_registered_name() -> None:
    assert ExactMatch.name == "exact_match"
