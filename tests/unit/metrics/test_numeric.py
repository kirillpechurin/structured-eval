"""Numeric — tolerant numeric equality over a lenient parser.

Strips currency/separators, honors accounting notation ``(123) → −123``, and
supports relative / absolute tolerance bands. Booleans are not numbers.
"""

import pytest

from structured_eval import Numeric
from structured_eval.metrics.numeric import NumericMode

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("metric", "actual", "expected", "score"),
    [
        (Numeric(tolerance=0.01), 100.5, 100.0, 1.0),  # within relative band
        (Numeric(tolerance=0.01), 110.0, 100.0, 0.0),  # outside relative band
        (Numeric(tolerance=2, mode=NumericMode.ABSOLUTE), 101, 100, 1.0),  # |1| <= 2
        (Numeric(tolerance=2, mode=NumericMode.ABSOLUTE), 105, 100, 0.0),  # |5| > 2
        (Numeric(), 0, 0, 1.0),  # expected zero, exact
        (Numeric(), 1, 0, 0.0),  # expected zero, off
    ],
    ids=["rel-in", "rel-out", "abs-in", "abs-out", "zero-exact", "zero-off"],
)
def test_tolerance_bands(metric, actual, expected, score) -> None:
    assert metric.score(actual, expected) == score


@pytest.mark.parametrize(
    ("actual", "expected", "score"),
    [
        ("100", 100, 1.0),  # numeric string coerced
        ("abc", 100, 0.0),  # non-numeric → 0.0
        ("$1,234.50", 1234.50, 1.0),  # currency + separators stripped
        ("(123)", -123, 1.0),  # accounting notation → negative
        ("(123)", 123, 0.0),  # ... and it really is negative
        ("1e3", 1000, 1.0),  # scientific notation
        ("1.5e-3", 0.0015, 1.0),
        ("50%", 50, 1.0),  # "%" dropped, number unchanged (not 0.5)
        ("50%", 0.5, 0.0),
        (True, 1, 0.0),  # bool is not numeric
    ],
    ids=[
        "str-coerce",
        "non-numeric",
        "currency",
        "accounting-neg",
        "accounting-sign",
        "sci",
        "sci-small",
        "percent-dropped",
        "percent-not-fraction",
        "bool-not-number",
    ],
)
def test_lenient_parsing(actual, expected, score) -> None:
    assert Numeric(tolerance=0).score(actual, expected) == score


def test_explicit_bands_union() -> None:
    # within the absolute band but not the relative one → still a match.
    metric = Numeric(relative_tolerance=0.001, absolute_tolerance=5)
    assert metric.score(103, 100) == 1.0  # |3| <= 5
    assert metric.score(200, 100) == 0.0  # outside both


def test_explicit_band_overrides_tolerance() -> None:
    metric = Numeric(tolerance=0, relative_tolerance=0.1)
    assert metric.score(109, 100) == 1.0
