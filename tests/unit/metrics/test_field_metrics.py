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
    Numeric,
    NumericCloseness,
    Presence,
    RegexMatch,
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


class TestRegexMatch:
    def test_case_and_whitespace(self):
        assert RegexMatch().score("  Acme   Corp ", "acme corp") == 1.0

    def test_default_collapses_spaces(self):
        assert RegexMatch().score("a\tb", "a b") == 1.0

    def test_still_distinguishes_values(self):
        assert RegexMatch().score("Acme", "Globex") == 0.0

    def test_custom_pattern_drops_punctuation(self):
        m = RegexMatch(pattern=r"[^\w\s]", repl="")
        assert m.score("hello, world!", "hello world") == 1.0

    def test_string_only_non_str_is_zero(self):
        # A string-only metric: non-str on either side scores 0.0, never coerced.
        assert RegexMatch().score(12, 12.0) == 0.0
        assert RegexMatch().score(None, None) == 0.0
        assert RegexMatch().score(None, "none") == 0.0
        assert RegexMatch().score("12", 12) == 0.0

    def test_lower_and_strip_flags(self):
        assert RegexMatch(lower=False).score("Acme", "acme") == 0.0
        assert RegexMatch(strip=False).score(" acme ", "acme") == 0.0


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

    def test_currency_and_separators_stripped(self):
        assert Numeric(tolerance=0).score("$1,234.50", 1234.50) == 1.0
        assert Numeric(tolerance=0).score("1 234,5".replace(",", ""), 12345) == 1.0

    def test_accounting_notation_negative(self):
        assert Numeric(tolerance=0).score("(123)", -123) == 1.0
        assert Numeric(tolerance=0).score("(123)", 123) == 0.0

    def test_scientific_notation_supported(self):
        assert Numeric(tolerance=0).score("1e3", 1000) == 1.0
        assert Numeric(tolerance=0).score("1.5e-3", 0.0015) == 1.0

    def test_percent_sign_stripped_not_interpreted(self):
        # Deliberate: "%" is dropped, the number is unchanged ("50%" → 50, not 0.5).
        assert Numeric(tolerance=0).score("50%", 50) == 1.0
        assert Numeric(tolerance=0).score("50%", 0.5) == 0.0

    def test_bool_is_not_numeric(self):
        assert Numeric().score(True, 1) == 0.0

    def test_explicit_relative_and_absolute_bands(self):
        # within absolute band but not relative
        m = Numeric(relative_tolerance=0.001, absolute_tolerance=5)
        assert m.score(103, 100) == 1.0  # |3| <= 5
        assert m.score(200, 100) == 0.0  # outside both

    def test_explicit_band_overrides_tolerance(self):
        m = Numeric(tolerance=0, relative_tolerance=0.1)
        assert m.score(109, 100) == 1.0


class TestNumericCloseness:
    def test_graded_ratio(self):
        # ratio similarity = min/max for same-sign values
        assert NumericCloseness().score(10, 12) == pytest.approx(10 / 12)
        assert NumericCloseness().score(100, 110) == pytest.approx(100 / 110)

    def test_equal_is_one(self):
        assert NumericCloseness().score(42, 42) == 1.0
        assert NumericCloseness().score(0, 0) == 1.0

    def test_opposite_signs_clamped_to_zero(self):
        assert NumericCloseness().score(5, -5) == 0.0

    def test_parses_numeric_strings(self):
        # shares Numeric's lenient parser → strings are graded, not collapsed
        assert NumericCloseness().score("100", 100) == 1.0
        assert NumericCloseness().score("100", "110") == pytest.approx(100 / 110)
        assert NumericCloseness().score("$120", 100) == pytest.approx(100 / 120)

    def test_non_number_is_zero(self):
        # applies only to numbers: any non-numeric side → 0.0, no equality fallback
        assert NumericCloseness().score(True, 1) == 0.0  # bool is not numeric
        assert NumericCloseness().score("abc", 100) == 0.0
        assert NumericCloseness().score(None, None) == 0.0  # null → inapplicable
        assert NumericCloseness().score("abc", "abc") == 0.0


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

    def test_multiset_repeated_tokens(self):
        # SQuAD-style: repeats count with multiplicity, not collapsed to a set.
        # "the the cat" (3 toks) vs "the cat" (2): common {the:1, cat:1} = 2
        # p = 2/3, r = 2/2 → f1 = 2*(2/3)*1 / (2/3 + 1) = 0.8
        assert TokenF1().score("the the cat", "the cat") == pytest.approx(0.8)

    def test_string_only_non_str_is_zero(self):
        assert TokenF1().score(None, None) == 0.0
        assert TokenF1().score(None, "none") == 0.0
        assert TokenF1().score(123, 123.0) == 0.0


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

    def test_ratio_is_order_sensitive(self):
        # plain ratio, unlike token_sort, is hurt by reordering
        assert Fuzzy(method="ratio").score("world hello", "hello world") < 1.0

    def test_normalize_disabled_keeps_case(self):
        assert Fuzzy(normalize=False).score("ACME", "acme") < 1.0
        assert Fuzzy(normalize=True).score("ACME", "acme") == 1.0

    def test_string_only_non_str_is_zero(self):
        assert Fuzzy().score(None, None) == 0.0
        assert Fuzzy().score(None, "none") == 0.0
        assert Fuzzy().score(123, 123.0) == 0.0


class TestLevenshtein:
    def test_is_ratio_alias(self):
        from structured_eval import Levenshtein

        assert Levenshtein.name == "levenshtein"
        assert Levenshtein().score("kitten", "kitten") == 1.0
        # matches Fuzzy ratio exactly
        assert Levenshtein().score("kitten", "sitting") == Fuzzy(method="ratio").score(
            "kitten", "sitting"
        )


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
