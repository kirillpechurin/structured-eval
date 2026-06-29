"""The rule_pass_rate package: Rule DSL (dsl.py) + RuleProcessor (engine.py).

One cohesive unit — comparisons, JSONPath arithmetic, ``in_``, custom rules,
failure paths, and pass-rate aggregation.
"""

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
    with pytest.raises(ValueError):
        Rule("$.total").evaluate(DOC)


def test_name_reflects_comparison() -> None:
    assert Rule("$.total").gt(0).name == "$.total gt 0"


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
