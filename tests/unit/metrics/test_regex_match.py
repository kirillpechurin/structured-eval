"""RegexMatch — equality after optional lower/strip + regex substitution.

String-only: a non-str on either side scores 0.0, never coerced.
"""

from typing import Any

import pytest

from structured_eval.metrics import RegexMatch

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("actual", "expected", "score"),
    [
        ("  Acme   Corp ", "acme corp", 1.0),  # default lower+strip+collapse
        ("a\tb", "a b", 1.0),  # whitespace collapsed
        ("Acme", "Globex", 0.0),  # genuinely different
    ],
    ids=["case-and-whitespace", "collapse-tabs", "distinct-values"],
)
def test_default_normalization(actual: Any, expected: Any, score: Any) -> None:
    assert RegexMatch().score(actual, expected) == score


def test_custom_pattern_drops_punctuation() -> None:
    metric = RegexMatch(pattern=r"[^\w\s]", repl="")
    assert metric.score("hello, world!", "hello world") == 1.0


@pytest.mark.parametrize(
    ("actual", "expected"),
    [(12, 12.0), (None, None), (None, "none"), ("12", 12)],
    ids=["int-int", "null-null", "null-str", "str-int"],
)
def test_string_only_non_str_is_zero(actual: Any, expected: Any) -> None:
    assert RegexMatch().score(actual, expected) == 0.0


@pytest.mark.parametrize(
    ("metric", "actual", "expected", "score"),
    [
        (RegexMatch(lower=False), "Acme", "acme", 0.0),  # case kept → differ
        (RegexMatch(strip=False), " acme ", "acme", 0.0),  # padding kept → differ
    ],
    ids=["lower-off", "strip-off"],
)
def test_flags_disable_normalization(
    metric: Any, actual: Any, expected: Any, score: Any
) -> None:
    assert metric.score(actual, expected) == score
