"""Unit tests for the field (scalar) comparison metrics.

A field metric *is* the comparison: ``score(actual, expected) -> float`` is a
pure primitive, tested here without the engine. ``Presence`` overrides
``compute(node)`` instead, so it is exercised through a ScalarNode.
"""

from __future__ import annotations

import pytest

from structured_eval import (
    ExactMatch,
    Fuzzy,
    NormalizedMatch,
    Numeric,
    Presence,
    TokenF1,
    TypeMatch,
)
from structured_eval.model.nodes.scalar import ScalarNode

pytestmark = pytest.mark.unit


class TestExactMatch:
    def test_equal(self):
        assert ExactMatch().score("paid", "paid") == 1.0

    def test_unequal(self):
        assert ExactMatch().score("paid", "draft") == 0.0

    def test_type_sensitive(self):
        assert ExactMatch().score("100", 100) == 0.0

    def test_none_equals_none(self):
        assert ExactMatch().score(None, None) == 1.0

    def test_registered_name(self):
        assert ExactMatch.name == "exact_match"


class TestNormalizedMatch:
    def test_case_and_whitespace(self):
        assert NormalizedMatch().score("  Acme   Corp ", "acme corp") == 1.0

    def test_default_collapses_spaces(self):
        assert NormalizedMatch().score("a\tb", "a b") == 1.0

    def test_still_distinguishes_values(self):
        assert NormalizedMatch().score("Acme", "Globex") == 0.0

    def test_custom_pattern_drops_punctuation(self):
        m = NormalizedMatch(pattern=r"[^\w\s]", repl="")
        assert m.score("hello, world!", "hello world") == 1.0


class TestNumeric:
    def test_within_relative_tolerance(self):
        assert Numeric(tolerance=0.01).score(100.5, 100.0) == 1.0

    def test_outside_relative_tolerance(self):
        assert Numeric(tolerance=0.01).score(110.0, 100.0) == 0.0

    def test_absolute_mode(self):
        assert Numeric(tolerance=2, mode="absolute").score(101, 100) == 1.0
        assert Numeric(tolerance=2, mode="absolute").score(105, 100) == 0.0

    def test_expected_zero_relative(self):
        assert Numeric().score(0, 0) == 1.0
        assert Numeric().score(1, 0) == 0.0

    def test_string_number_coerced(self):
        assert Numeric(tolerance=0).score("100", 100) == 1.0

    def test_non_numeric_is_zero(self):
        assert Numeric().score("abc", 100) == 0.0


class TestTokenF1:
    def test_identical(self):
        assert TokenF1().score("the quick fox", "the quick fox") == 1.0

    def test_partial_overlap(self):
        # actual {the,quick,brown,fox}, expected {the,quick,fox}
        score = TokenF1().score("the quick brown fox", "the quick fox")
        # p = 3/4, r = 3/3 → f1 = 2*.75*1/(1.75)
        assert score == pytest.approx(2 * 0.75 * 1.0 / 1.75)

    def test_disjoint(self):
        assert TokenF1().score("alpha", "beta") == 0.0

    def test_both_empty(self):
        assert TokenF1().score("", "") == 1.0

    def test_one_empty(self):
        assert TokenF1().score("", "x") == 0.0

    def test_punctuation_ignored(self):
        assert TokenF1().score("hello, world.", "hello world") == 1.0


class TestTypeMatch:
    def test_same_type(self):
        assert TypeMatch().score(100, 200) == 1.0

    def test_string_vs_number(self):
        assert TypeMatch().score("100", 100) == 0.0

    def test_bool_is_not_number(self):
        assert TypeMatch().score(True, 1) == 0.0

    def test_null_matches_null(self):
        assert TypeMatch().score(None, None) == 1.0

    def test_list_and_dict(self):
        assert TypeMatch().score([1], [2, 3]) == 1.0
        assert TypeMatch().score({"a": 1}, {}) == 1.0


class TestFuzzy:
    def test_identical(self):
        assert Fuzzy().score("hello world", "hello world") == 1.0

    def test_token_sort(self):
        # token_sort_ratio is order-insensitive
        assert Fuzzy().score("world hello", "hello world") == pytest.approx(1.0)

    def test_partial(self):
        score = Fuzzy().score("Acme Corporation", "Acme Corp")
        assert 0.0 < score < 1.0


class TestPresence:
    def _node(self, actual, context_factory):
        ctx = context_factory({"x": actual}, {"x": actual})
        return ScalarNode(path="x", context=ctx)

    def test_present(self, context_factory):
        node = self._node("value", context_factory)
        assert Presence().compute(node) == 1.0

    def test_absent(self, context_factory):
        ctx = context_factory({}, {"x": 1})
        node = ScalarNode(path="x", context=ctx)
        assert Presence().compute(node) == 0.0

    def test_ignores_expected(self, context_factory):
        # present in actual, even if it mismatches expected → still 1.0
        ctx = context_factory({"x": "a"}, {"x": "b"})
        node = ScalarNode(path="x", context=ctx)
        assert Presence().compute(node) == 1.0
