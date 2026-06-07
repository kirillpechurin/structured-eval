import pytest

from structured_eval.core.result import RuleResult
from structured_eval.rules.dsl import Rule
from structured_eval.rules.engine import run_rules

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


# ── Basic comparisons ─────────────────────────────────────────────────────────


class TestRuleEq:
    def test_pass(self):
        r = Rule("$.status").eq("paid").evaluate(DOC)
        assert r.passed is True
        assert r.message == ""

    def test_fail(self):
        r = Rule("$.status").eq("pending").evaluate(DOC)
        assert r.passed is False
        assert "$.status" in r.message

    def test_numeric(self):
        assert Rule("$.total").eq(110.0).evaluate(DOC).passed is True
        assert Rule("$.total").eq(999.0).evaluate(DOC).passed is False


class TestRuleOrdering:
    def test_lt_pass(self):
        assert Rule("$.subtotal").lt(200.0).evaluate(DOC).passed is True

    def test_lt_fail(self):
        assert Rule("$.total").lt(50.0).evaluate(DOC).passed is False

    def test_gt_pass(self):
        assert Rule("$.total").gt(100.0).evaluate(DOC).passed is True

    def test_gt_fail(self):
        assert Rule("$.total").gt(200.0).evaluate(DOC).passed is False

    def test_lte_equal(self):
        assert Rule("$.total").lte(110.0).evaluate(DOC).passed is True

    def test_lte_less(self):
        assert Rule("$.total").lte(200.0).evaluate(DOC).passed is True

    def test_lte_fail(self):
        assert Rule("$.total").lte(50.0).evaluate(DOC).passed is False

    def test_gte_equal(self):
        assert Rule("$.total").gte(110.0).evaluate(DOC).passed is True

    def test_gte_greater(self):
        assert Rule("$.total").gte(50.0).evaluate(DOC).passed is True

    def test_gte_fail(self):
        assert Rule("$.total").gte(200.0).evaluate(DOC).passed is False


class TestRuleIn:
    def test_pass(self):
        assert Rule("$.currency").in_(["USD", "EUR", "GBP"]).evaluate(DOC).passed is True

    def test_fail(self):
        assert Rule("$.currency").in_(["EUR", "GBP"]).evaluate(DOC).passed is False


# ── RHS as JSONPath ───────────────────────────────────────────────────────────


class TestRuleRhsPath:
    def test_eq_two_paths(self):
        # $.total != $.subtotal (110 != 100)
        assert Rule("$.total").eq("$.subtotal").evaluate(DOC).passed is False

    def test_eq_matching_paths(self):
        doc = {**DOC, "copy": 110.0}
        assert Rule("$.total").eq("$.copy").evaluate(doc).passed is True

    def test_gt_path(self):
        assert Rule("$.total").gt("$.subtotal").evaluate(DOC).passed is True


# ── RHS as arithmetic expression ──────────────────────────────────────────────


class TestRuleArithmetic:
    def test_addition(self):
        # 110.0 == 100.0 + 10.0
        assert Rule("$.total").eq("$.subtotal + $.tax").evaluate(DOC).passed is True

    def test_addition_fail(self):
        doc = {**DOC, "tax": 5.0}
        assert Rule("$.total").eq("$.subtotal + $.tax").evaluate(doc).passed is False

    def test_subtraction(self):
        # subtotal == total - tax
        assert Rule("$.subtotal").eq("$.total - $.tax").evaluate(DOC).passed is True

    def test_multiplication(self):
        doc = {"price": 10.0, "qty": 3, "line_total": 30.0}
        assert Rule("$.line_total").eq("$.price * $.qty").evaluate(doc).passed is True

    def test_division(self):
        doc = {"total": 100.0, "count": 4, "avg": 25.0}
        assert Rule("$.avg").eq("$.total / $.count").evaluate(doc).passed is True


# ── Rule.custom ───────────────────────────────────────────────────────────────


class TestRuleCustom:
    def test_pass(self):
        rule = Rule.custom(lambda doc: doc["amount"] > 0, name="positive_amount")
        r = rule.evaluate(DOC)
        assert r.passed is True
        assert r.name == "positive_amount"

    def test_fail(self):
        rule = Rule.custom(lambda doc: doc["amount"] < 0, name="negative_amount")
        r = rule.evaluate(DOC)
        assert r.passed is False

    def test_exception_becomes_failure(self):
        rule = Rule.custom(lambda doc: 1 / 0, name="boom")  # type: ignore[arg-type]
        r = rule.evaluate(DOC)
        assert r.passed is False
        assert "division by zero" in r.message

    def test_default_name(self):
        rule = Rule.custom(lambda doc: True)
        assert rule.name == "custom"


# ── Rule name ─────────────────────────────────────────────────────────────────


class TestRuleName:
    def test_explicit_name(self):
        r = Rule("$.status", name="status_check").eq("paid")
        assert r.name == "status_check"

    def test_auto_name_string_rhs(self):
        r = Rule("$.status").eq("paid")
        assert r.name == "$.status eq paid"

    def test_auto_name_numeric_rhs(self):
        r = Rule("$.total").gte(100)
        assert r.name == "$.total gte 100"

    def test_auto_name_expr_rhs(self):
        r = Rule("$.total").eq("$.subtotal + $.tax")
        assert r.name == "$.total eq $.subtotal + $.tax"


# ── Missing path ──────────────────────────────────────────────────────────────


class TestRuleMissingPath:
    def test_lhs_missing(self):
        r = Rule("$.nonexistent").eq("x").evaluate(DOC)
        assert r.passed is False
        assert "nonexistent" in r.message

    def test_rhs_path_missing(self):
        r = Rule("$.total").eq("$.nonexistent").evaluate(DOC)
        assert r.passed is False


# ── Unbound rule ──────────────────────────────────────────────────────────────


def test_unbound_rule_raises():
    with pytest.raises(ValueError, match="no comparison"):
        Rule("$.status").evaluate(DOC)


# ── run_rules engine ──────────────────────────────────────────────────────────


class TestRunRules:
    def test_all_pass(self):
        rules = [Rule("$.status").eq("paid"), Rule("$.total").gte(0)]
        results, rate = run_rules(rules, DOC)
        assert len(results) == 2
        assert rate == 1.0

    def test_partial_pass(self):
        rules = [Rule("$.status").eq("paid"), Rule("$.total").eq(999)]
        results, rate = run_rules(rules, DOC)
        assert rate == pytest.approx(0.5)
        assert results[0].passed is True
        assert results[1].passed is False

    def test_all_fail(self):
        rules = [Rule("$.status").eq("pending"), Rule("$.total").eq(0)]
        _, rate = run_rules(rules, DOC)
        assert rate == 0.0

    def test_empty_rules(self):
        results, rate = run_rules([], DOC)
        assert results == []
        assert rate == 1.0

    def test_custom_rule_in_engine(self):
        rules = [Rule.custom(lambda d: d["amount"] > 0, name="pos")]
        results, rate = run_rules(rules, DOC)
        assert rate == 1.0
        assert results[0].name == "pos"


# ── Integration: evaluate() with rules ───────────────────────────────────────


def test_evaluate_with_rules():
    from structured_eval import EvalConfig, evaluate

    actual = {"status": "paid", "total": 110.0, "subtotal": 100.0, "tax": 10.0}
    expected = {"status": "paid", "total": 110.0, "subtotal": 100.0, "tax": 10.0}

    report = evaluate(
        actual,
        expected,
        config=EvalConfig(
            rules=[
                Rule("$.status").eq("paid"),
                Rule("$.total").eq("$.subtotal + $.tax"),
                Rule("$.total").gte(0),
            ]
        ),
        detailed=True,
    )

    assert report.rule_pass_rate == 1.0
    assert len(report.rule_results) == 3
    assert all(r.passed for r in report.rule_results)


def test_evaluate_with_failing_rule():
    from structured_eval import EvalConfig, evaluate

    actual = {"status": "draft", "total": 50.0}
    expected = {"status": "draft", "total": 50.0}

    report = evaluate(
        actual,
        expected,
        config=EvalConfig(rules=[Rule("$.status").eq("paid")]),
        detailed=True,
    )

    assert report.rule_pass_rate == pytest.approx(0.0)
    assert report.rule_results[0].passed is False
