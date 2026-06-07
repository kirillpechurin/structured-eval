import pytest
from pydantic import BaseModel

from structured_eval import EvalConfig, evaluate
from structured_eval.rules.dsl import Rule


class Invoice(BaseModel):
    id: str
    total: float
    status: str


# ── f1/precision/recall are None without expected ─────────────────────────────


def test_core_metrics_none_without_expected():
    report = evaluate({"id": "1", "total": 100.0}, config=EvalConfig(), detailed=True)
    assert report.f1 is None
    assert report.precision is None
    assert report.recall is None
    assert report.perfect is None
    assert report.field_scores == {}
    assert report.type_error_rate is None


# ── Schema-only with json_schema ──────────────────────────────────────────────


def test_schema_only_valid():
    actual = {"id": "INV-1", "total": 100.0, "status": "paid"}
    report = evaluate(actual, config=EvalConfig(json_schema=Invoice), detailed=True)
    assert report.f1 is None
    assert report.schema_valid is True
    assert report.coverage_score == pytest.approx(1.0)


def test_schema_only_invalid():
    actual = {"id": "INV-1", "total": "not-a-float"}
    report = evaluate(actual, config=EvalConfig(json_schema=Invoice), detailed=True)
    assert report.f1 is None
    assert report.schema_valid is False


def test_schema_only_coverage_partial():
    actual = {"id": "INV-1", "total": None, "status": "paid"}
    report = evaluate(actual, config=EvalConfig(json_schema=Invoice), detailed=True)
    # total is null → not covered; id and status are covered → 2/3
    assert report.coverage_score == pytest.approx(2 / 3)


# ── Schema-only with rules ────────────────────────────────────────────────────


def test_rules_only_all_pass():
    actual = {"status": "paid", "total": 110.0, "subtotal": 100.0, "tax": 10.0}
    report = evaluate(
        actual,
        config=EvalConfig(rules=[
            Rule("$.status").eq("paid"),
            Rule("$.total").eq("$.subtotal + $.tax"),
        ]),
        detailed=True,
    )
    assert report.f1 is None
    assert report.rule_pass_rate == 1.0
    assert len(report.rule_results) == 2


def test_rules_only_partial_pass():
    actual = {"status": "draft", "total": 50.0}
    report = evaluate(
        actual,
        config=EvalConfig(rules=[
            Rule("$.status").eq("paid"),
            Rule("$.total").gt(0),
        ]),
        detailed=True,
    )
    assert report.f1 is None
    assert report.rule_pass_rate == pytest.approx(0.5)


# ── schema + rules combined ───────────────────────────────────────────────────


def test_schema_and_rules():
    actual = {"id": "INV-1", "total": 100.0, "status": "paid"}
    report = evaluate(
        actual,
        config=EvalConfig(
            json_schema=Invoice,
            rules=[Rule("$.total").gt(0), Rule("$.status").in_(["paid", "draft"])],
        ),
        detailed=True,
    )
    assert report.f1 is None
    assert report.schema_valid is True
    assert report.rule_pass_rate == 1.0


# ── score property ────────────────────────────────────────────────────────────


def test_score_returns_f1_when_expected_provided():
    actual = {"id": "1", "name": "Alice"}
    expected = {"id": "1", "name": "Alice"}
    report = evaluate(actual, expected, detailed=True)
    assert report.f1 == 1.0
    assert report.score == report.f1


def test_score_falls_back_to_rule_pass_rate():
    actual = {"total": 100.0, "subtotal": 90.0, "tax": 10.0}
    report = evaluate(
        actual,
        config=EvalConfig(rules=[Rule("$.total").eq("$.subtotal + $.tax")]),
        detailed=True,
    )
    assert report.f1 is None
    assert report.rule_pass_rate == 1.0
    assert report.score == 1.0


def test_score_falls_back_to_coverage_score():
    actual = {"id": "1", "total": 100.0, "status": "paid"}
    report = evaluate(actual, config=EvalConfig(json_schema=Invoice), detailed=True)
    assert report.f1 is None
    assert report.rule_pass_rate is None
    assert report.score == report.coverage_score


def test_score_none_when_nothing_computable():
    actual = {"x": 1}
    report = evaluate(actual, config=EvalConfig(), detailed=True)
    assert report.score is None


# ── perfect flag ──────────────────────────────────────────────────────────────


def test_perfect_true():
    actual = {"id": "1", "name": "Alice"}
    expected = {"id": "1", "name": "Alice"}
    report = evaluate(actual, expected, detailed=True)
    assert report.perfect is True


def test_perfect_false():
    actual = {"id": "1", "name": "Bob"}
    expected = {"id": "1", "name": "Alice"}
    report = evaluate(actual, expected, detailed=True)
    assert report.perfect is False


def test_perfect_none_without_expected():
    report = evaluate({"x": 1}, config=EvalConfig(), detailed=True)
    assert report.perfect is None


# ── path metrics absent without expected ─────────────────────────────────────


def test_path_metrics_none_without_expected():
    actual = {"id": "1", "total": 100.0}
    report = evaluate(actual, config=EvalConfig(json_schema=Invoice), detailed=True)
    assert report.path_recall is None
    assert report.path_precision is None
