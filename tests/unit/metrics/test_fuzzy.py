"""Fuzzy (rapidfuzz) and its thin alias Levenshtein.

``Levenshtein`` *is* ``Fuzzy(method=RATIO)`` — one cohesive unit, so both live
here. Fuzzy is string-only.
"""

from typing import Any

import pytest

from structured_eval.metrics import Fuzzy, Levenshtein
from structured_eval.metrics.fuzzy import FuzzyMethod

pytestmark = pytest.mark.unit


def test_identical_is_one() -> None:
    assert Fuzzy().score("hello world", "hello world") == 1.0


def test_null() -> None:
    assert Fuzzy().score(None, None) == 1.0


def test_token_sort_is_order_insensitive() -> None:
    # default token_sort_ratio ignores word order
    assert Fuzzy().score("world hello", "hello world") == pytest.approx(1.0)


def test_ratio_is_order_sensitive() -> None:
    # plain ratio, unlike token_sort, is hurt by reordering
    assert Fuzzy(method=FuzzyMethod.RATIO).score("world hello", "hello world") < 1.0


def test_partial_match_is_between() -> None:
    assert 0.0 < Fuzzy().score("Acme Corporation", "Acme Corp") < 1.0


# ``ignore_case`` and ``ignore_whitespace`` are independent: each folds only its
# own dimension, leaving the other significant. ``ignore_whitespace`` collapses
# whitespace runs to one space and trims the ends. Pinned to ``ratio`` — the
# default ``token_sort_ratio`` normalizes whitespace itself.
@pytest.mark.parametrize(
    ("kwargs", "actual", "expected", "predicate"),
    [
        ({"ignore_case": True}, "ACME", "acme", lambda s: s == 1.0),  # case folded
        ({"ignore_case": False}, "ACME", "acme", lambda s: s < 1.0),  # case sig.
        # case-only: whitespace stays significant
        (
            {"ignore_case": True, "ignore_whitespace": False},
            " a  b ",
            "a b",
            lambda s: s < 1.0,
        ),
        # whitespace-only: case stays significant
        (
            {"ignore_case": False, "ignore_whitespace": True},
            "ACME",
            "acme",
            lambda s: s < 1.0,
        ),
        # whitespace-only: runs collapsed and ends trimmed → identical
        (
            {"ignore_case": False, "ignore_whitespace": True},
            " a  b ",
            "a b",
            lambda s: s == 1.0,
        ),
    ],
    ids=[
        "case-on",
        "case-off",
        "case-only-ws-sig",
        "ws-only-case-sig",
        "ws-only-collapses",
    ],
)
def test_normalization_flags_are_independent(
    kwargs: Any, actual: Any, expected: Any, predicate: Any
) -> None:
    assert predicate(Fuzzy(method=FuzzyMethod.RATIO, **kwargs).score(actual, expected))


@pytest.mark.parametrize(
    ("actual", "expected"),
    [(None, "none"), (123, 123.0)],
    ids=["null-str", "int-float"],
)
def test_string_only_non_str_is_zero(actual: Any, expected: Any) -> None:
    assert Fuzzy().score(actual, expected) == 0.0


# ── Levenshtein alias ─────────────────────────────────────────────────────────


def test_levenshtein_is_registered() -> None:
    assert Levenshtein.name == "levenshtein"


def test_levenshtein_matches_fuzzy_ratio() -> None:
    assert Levenshtein().score("kitten", "kitten") == 1.0
    assert Levenshtein().score("kitten", "sitting") == Fuzzy(
        method=FuzzyMethod.RATIO
    ).score("kitten", "sitting")


@pytest.mark.parametrize(
    ("kwargs", "actual"),
    [({"ignore_case": False}, "ACME"), ({"ignore_whitespace": False}, "  acme  ")],
    ids=["case", "whitespace"],
)
def test_levenshtein_inherits_normalization_options(kwargs: Any, actual: Any) -> None:
    assert Levenshtein(**kwargs).score(actual, "acme") < 1.0
