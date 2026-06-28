"""NumericCloseness — graded ratio similarity ``min/max`` for same-sign numbers.

Numbers only: any non-number side scores 0.0, with no equality fallback. Shares
``Numeric``'s lenient parser, so numeric strings are graded rather than collapsed.
"""

import pytest

from structured_eval import NumericCloseness

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("actual", "expected", "score"),
    [
        (10, 12, 10 / 12),  # graded ratio
        (100, 110, 100 / 110),
        (42, 42, 1.0),  # equal → 1.0
        (0, 0, 1.0),
        (5, -5, 0.0),  # opposite signs clamp to 0
        ("100", 100, 1.0),  # numeric string parsed
        ("100", "110", 100 / 110),
        ("$120", 100, 100 / 120),  # currency parsed then graded
    ],
    ids=[
        "ratio",
        "ratio-100",
        "equal",
        "zero",
        "opposite-signs",
        "str-exact",
        "str-graded",
        "currency-graded",
    ],
)
def test_graded_similarity(actual, expected, score):
    assert NumericCloseness().score(actual, expected) == pytest.approx(score)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (True, 1),  # bool is not numeric
        ("abc", 100),
        (None, None),  # null → inapplicable
        ("abc", "abc"),  # no equality fallback for non-numbers
    ],
    ids=["bool", "non-numeric", "null", "equal-non-number"],
)
def test_non_number_is_zero(actual, expected):
    assert NumericCloseness().score(actual, expected) == 0.0
