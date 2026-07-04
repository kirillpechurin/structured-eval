"""ExponentialNumericScore — ``exp(-|a-e|/scale)`` similarity for numbers.

Numbers only (shares ``Numeric``'s lenient parser); any non-number side scores
0.0. ``scale`` controls the decay; it must be > 0.
"""

import math
from typing import Any

import pytest

from structured_eval.metrics import ExponentialNumericScore

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("actual", "expected", "scale", "score"),
    [
        (5, 5, 1.0, 1.0),  # exact match → 1.0
        (10, 12, 5.0, math.exp(-2 / 5)),  # decay on absolute error
        (10, 12, 1.0, math.exp(-2)),  # smaller scale → steeper decay
        ("100", 100, 1.0, 1.0),  # numeric string parsed
        ("$120", 100, 50.0, math.exp(-20 / 50)),  # currency parsed then scored
    ],
    ids=["exact", "scale-5", "scale-1", "str-exact", "currency"],
)
def test_exponential_score(
    actual: Any, expected: Any, scale: float, score: Any
) -> None:
    assert ExponentialNumericScore(scale=scale).score(
        actual, expected
    ) == pytest.approx(score)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (True, 1),  # bool is not numeric
        ("abc", 100),
        (None, None),  # null → inapplicable
    ],
    ids=["bool", "non-numeric", "null"],
)
def test_non_number_is_zero(actual: Any, expected: Any) -> None:
    assert ExponentialNumericScore().score(actual, expected) == 0.0


@pytest.mark.parametrize("scale", [0.0, -1.0], ids=["zero", "negative"])
def test_non_positive_scale_rejected(scale: float) -> None:
    with pytest.raises(ValueError, match="scale must be greater than 0"):
        ExponentialNumericScore(scale=scale)
