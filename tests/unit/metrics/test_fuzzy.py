"""Fuzzy (rapidfuzz) and its thin alias Levenshtein.

``Levenshtein`` *is* ``Fuzzy(method=RATIO)`` — one cohesive unit, so both live
here. Fuzzy is string-only.
"""

import pytest

from structured_eval import Fuzzy, Levenshtein

pytestmark = pytest.mark.unit


def test_identical_is_one():
    assert Fuzzy().score("hello world", "hello world") == 1.0


def test_token_sort_is_order_insensitive():
    # default token_sort_ratio ignores word order
    assert Fuzzy().score("world hello", "hello world") == pytest.approx(1.0)


def test_ratio_is_order_sensitive():
    # plain ratio, unlike token_sort, is hurt by reordering
    assert Fuzzy(method="ratio").score("world hello", "hello world") < 1.0


def test_partial_match_is_between():
    assert 0.0 < Fuzzy().score("Acme Corporation", "Acme Corp") < 1.0


@pytest.mark.parametrize(
    ("normalize", "predicate"),
    [(False, lambda s: s < 1.0), (True, lambda s: s == 1.0)],
    ids=["normalize-off", "normalize-on"],
)
def test_normalize_controls_case_folding(normalize, predicate):
    assert predicate(Fuzzy(normalize=normalize).score("ACME", "acme"))


@pytest.mark.parametrize(
    ("actual", "expected"),
    [(None, None), (None, "none"), (123, 123.0)],
    ids=["null-null", "null-str", "int-float"],
)
def test_string_only_non_str_is_zero(actual, expected):
    assert Fuzzy().score(actual, expected) == 0.0


# ── Levenshtein alias ─────────────────────────────────────────────────────────


def test_levenshtein_is_registered():
    assert Levenshtein.name == "levenshtein"


def test_levenshtein_matches_fuzzy_ratio():
    assert Levenshtein().score("kitten", "kitten") == 1.0
    assert Levenshtein().score("kitten", "sitting") == Fuzzy(method="ratio").score(
        "kitten", "sitting"
    )
