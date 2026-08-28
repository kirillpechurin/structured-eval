"""The rule_pass_rate package: Rule DSL (dsl.py) + RuleProcessor (engine.py).

One cohesive unit — comparisons, JSONPath arithmetic, `in_`, custom rules,
failure paths, and pass-rate aggregation.
"""

import sys
from typing import Any

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


def _passed(rule: Rule, doc: dict[str, Any] = DOC) -> bool:
    """Whether the rule holds on the document."""
    return bool(rule.evaluate(doc).passed)


# ── comparisons ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("rule", "ok"),
    [
        (Rule("$.status").eq("paid"), True),
        (Rule("$.status").eq("draft"), False),
        (Rule("$.total").gt(0), True),
        (Rule("$.total").gte(110.0), True),
        (Rule("$.total").lt(200), True),
        (Rule("$.total").lte(110.0), True),
        (Rule("$.total").lt(0), False),
        (Rule("$.currency").in_(["USD", "EUR"]), True),
        (Rule("$.currency").in_(["GBP"]), False),
        (Rule("$.nested.value").eq(7), True),
    ],
    ids=[
        "eq-ok",
        "eq-no",
        "gt",
        "gte",
        "lt",
        "lte",
        "lt-no",
        "in-ok",
        "in-no",
        "nested",
    ],
)
def test_comparisons(rule: Any, ok: Any) -> None:
    assert _passed(rule) is ok


# ── JSONPath arithmetic ──────────────────────────────────────────────────────


def test_path_arithmetic_lhs() -> None:
    assert _passed(Rule("$.total").eq("$.subtotal + $.tax"))


def test_arithmetic_violation() -> None:
    bad = {"total": 999.0, "subtotal": 100.0, "tax": 10.0}
    assert not _passed(Rule("$.total").eq("$.subtotal + $.tax"), bad)


def test_path_on_rhs() -> None:
    assert _passed(Rule("$.subtotal").lt("$.total"))


# ── custom rules ─────────────────────────────────────────────────────────────


def test_custom_pass_with_name() -> None:
    result = Rule.custom(lambda d: d["amount"] > 0, name="positive").evaluate(DOC)
    assert result.passed
    assert result.name == "positive"


def test_custom_fail() -> None:
    assert not Rule.custom(lambda d: d["amount"] < 0).evaluate(DOC).passed


def test_custom_exception_is_failure() -> None:
    result = Rule.custom(lambda d: d["missing"]).evaluate(DOC)
    assert not result.passed
    assert result.message


# ── error paths ──────────────────────────────────────────────────────────────


def test_missing_path_fails_gracefully() -> None:
    result = Rule("$.nope").eq(1).evaluate(DOC)
    assert not result.passed
    assert "not found" in result.message


def test_no_comparison_raises() -> None:
    with pytest.raises(ValueError, match="comparison"):
        Rule("$.total").evaluate(DOC)


def test_a_path_on_the_right_that_is_missing_fails_gracefully() -> None:
    # The rule is well-formed; the *document* does not carry what it refers to,
    # so this is a failed rule rather than a crashed evaluation.
    result = Rule("$.total").eq("$.nope").evaluate(DOC)
    assert not result.passed
    assert result.message


def test_comparing_incomparable_types_fails_gracefully() -> None:
    result = Rule("$.status").lt(1).evaluate(DOC)  # "paid" < 1
    assert not result.passed
    assert result.message


@pytest.mark.parametrize(
    ("rule", "name"),
    [
        (Rule("$.total").gt(0), "$.total gt 0"),
        (Rule("$.total").eq("$.subtotal"), "$.total eq $.subtotal"),
        (Rule("$.total"), "$.total"),
        (Rule("$.total", name="totals add up").gt(0), "totals add up"),
    ],
    ids=["literal-rhs", "path-rhs", "no-comparison", "explicit-name"],
)
def test_a_rule_names_itself_after_what_it_checks(rule: Rule, name: str) -> None:
    # The name is what shows up in the report, so it has to read as the claim
    # being made — unless the caller supplied a better one.
    assert rule.name == name


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("$.total % 2", "Unsupported operator"),
        ("$.total > 1", "Unsupported expression node"),
    ],
    ids=["operator", "node"],
)
def test_arithmetic_is_restricted_to_arithmetic(expression: str, message: str) -> None:
    # The right-hand side is parsed, not `eval`-ed: only the four arithmetic
    # operators are honoured, and anything else is refused rather than run.
    # A malformed rule fails like any other, so one bad rule cannot take the
    # whole evaluation down with it.
    result = Rule("$.total").eq(expression).evaluate(DOC)
    assert not result.passed
    assert message in result.message


def test_a_negative_literal_is_arithmetic_too() -> None:
    assert _passed(Rule("$.total").eq("$.subtotal - -10"))


def test_a_missing_extra_is_reported_as_a_failed_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # jsonpath-ng is imported when a path is resolved, not at import time. The
    # rule cannot run without it, and the install hint travels in the message
    # rather than taking the whole evaluation down.
    monkeypatch.setitem(sys.modules, "jsonpath_ng", None)

    result = Rule("$.total").gt(0).evaluate(DOC)
    assert not result.passed
    assert "structured-eval[rules]" in result.message


def test_an_operator_the_dsl_cannot_produce_is_refused() -> None:
    # Unreachable through the fluent API — every comparison method sets a known
    # operator — so this guards the invariant rather than a user mistake.
    rule = Rule("$.total").eq(110.0)
    rule._op = "approximately"
    with pytest.raises(ValueError, match="Unknown operator"):
        rule._compare(1, 1)


# ── processor (pass-rate aggregation) ────────────────────────────────────────


def test_processor_all_pass() -> None:
    results, rate = RuleProcessor().run(
        [Rule("$.status").eq("paid"), Rule("$.total").gt(0)], DOC
    )
    assert rate == 1.0
    assert len(results) == 2


def test_processor_partial() -> None:
    _results, rate = RuleProcessor().run(
        [Rule("$.status").eq("draft"), Rule("$.total").gt(0)], DOC
    )
    assert rate == pytest.approx(0.5)


def test_processor_empty_is_vacuous() -> None:
    results, rate = RuleProcessor().run([], DOC)
    assert rate == 1.0
    assert results == []
