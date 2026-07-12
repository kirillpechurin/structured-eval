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


@pytest.mark.parametrize(
    ("normalize", "predicate"),
    [(False, lambda s: s < 1.0), (True, lambda s: s == 1.0)],
    ids=["normalize-off", "normalize-on"],
)
def test_normalize_controls_case_folding(normalize: Any, predicate: Any) -> None:
    assert predicate(Fuzzy(normalize=normalize).score("ACME", "acme"))


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
