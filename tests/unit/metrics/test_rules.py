"""Unit tests for the Rule DSL and RuleProcessor.

Covers comparisons, JSONPath arithmetic, ``in_``, custom rules, failure paths
(missing path, no comparison) and pass-rate aggregation.
"""

from __future__ import annotations

import pytest

from structured_eval.metrics.rule_pass_rate.dsl import Rule
from structured_eval.metrics.rule_pass_rate.engine import RuleProcessor

pytestmark = pytest.mark.unit

DOC = {
    "id": "INV-001",
    "status": "paid",
    "total": 110.0,
    "subtotal": 100.0,
    "tax": 10.0,
    "currency": "USD",
    "amount": 42,
    "nested": {"value": 7},
}


def _passed(rule, doc=DOC) -> bool:
    return rule.evaluate(doc).passed


class TestComparisons:
    def test_eq(self):
        assert _passed(Rule("$.status").eq("paid"))
        assert not _passed(Rule("$.status").eq("draft"))

    def test_gt_gte_lt_lte(self):
        assert _passed(Rule("$.total").gt(0))
        assert _passed(Rule("$.total").gte(110.0))
        assert _passed(Rule("$.total").lt(200))
        assert _passed(Rule("$.total").lte(110.0))
        assert not _passed(Rule("$.total").lt(0))

    def test_in(self):
        assert _passed(Rule("$.currency").in_(["USD", "EUR"]))
        assert not _passed(Rule("$.currency").in_(["GBP"]))

    def test_nested_path(self):
        assert _passed(Rule("$.nested.value").eq(7))


class TestArithmetic:
    def test_path_arithmetic(self):
        assert _passed(Rule("$.total").eq("$.subtotal + $.tax"))

    def test_arithmetic_failure(self):
        bad = {"total": 999.0, "subtotal": 100.0, "tax": 10.0}
        assert not _passed(Rule("$.total").eq("$.subtotal + $.tax"), bad)

    def test_path_rhs(self):
        assert _passed(Rule("$.subtotal").lt("$.total"))


class TestCustom:
    def test_custom_pass(self):
        rule = Rule.custom(lambda d: d["amount"] > 0, name="positive")
        result = rule.evaluate(DOC)
        assert result.passed
        assert result.name == "positive"

    def test_custom_fail(self):
        rule = Rule.custom(lambda d: d["amount"] < 0)
        assert not rule.evaluate(DOC).passed

    def test_custom_exception_is_failure(self):
        rule = Rule.custom(lambda d: d["missing"])
        result = rule.evaluate(DOC)
        assert not result.passed
        assert result.message


class TestErrors:
    def test_missing_path_fails_gracefully(self):
        result = Rule("$.nope").eq(1).evaluate(DOC)
        assert not result.passed
        assert "not found" in result.message

    def test_no_comparison_raises(self):
        with pytest.raises(ValueError):
            Rule("$.total").evaluate(DOC)

    def test_name_reflects_comparison(self):
        assert Rule("$.total").gt(0).name == "$.total gt 0"


class TestProcessor:
    def test_all_pass(self):
        rules = [Rule("$.status").eq("paid"), Rule("$.total").gt(0)]
        results, rate = RuleProcessor().run(rules, DOC)
        assert rate == 1.0
        assert len(results) == 2

    def test_partial(self):
        rules = [Rule("$.status").eq("draft"), Rule("$.total").gt(0)]
        _results, rate = RuleProcessor().run(rules, DOC)
        assert rate == pytest.approx(0.5)

    def test_empty_vacuous(self):
        results, rate = RuleProcessor().run([], DOC)
        assert rate == 1.0
        assert results == []
