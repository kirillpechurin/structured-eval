"""Schema-only / rules-only / faithfulness-only modes (no ``expected``).

Rewritten against the as-built API: metrics are instances in ``EvalConfig.metrics``
and results live in ``report.metrics`` / side channels — there is no ``detailed``
param and no ``report.f1`` shorthand.
"""

import pytest
from pydantic import BaseModel

from structured_eval import (
    CoverageLeafScore,
    EvalConfig,
    Rule,
    RulePassRate,
    SchemaValidity,
    evaluate,
)

pytestmark = pytest.mark.engine


class Invoice(BaseModel):
    id: str
    total: float
    status: str


class TestSchemaOnly:
    def test_valid(self):
        cfg = EvalConfig(metrics=[SchemaValidity(Invoice), CoverageLeafScore()])
        r = evaluate({"id": "INV-1", "total": 100.0, "status": "paid"}, config=cfg)
        assert r.metrics["schema_validity"].representative() == 1.0
        assert r.metrics["coverage_leaf_score"].representative() == pytest.approx(1.0)
        assert r.metrics["schema_validity"].root().extra["schema_errors"] == {
            "type_errors": [],
            "missing_required": [],
            "extra_fields": [],
        }

    def test_invalid(self):
        cfg = EvalConfig(metrics=[SchemaValidity(Invoice)])
        r = evaluate({"id": "INV-1", "total": "nope"}, config=cfg)
        assert r.metrics["schema_validity"].representative() == 0.0
        assert "total" in r.metrics["schema_validity"].root().extra["schema_errors"]["type_errors"]

    def test_coverage_partial_with_null(self):
        # total null → not covered; but CoverageLeafScore needs an expected reference
        cfg = EvalConfig(metrics=[CoverageLeafScore()])
        r = evaluate({"id": "1", "total": None}, {"id": "1", "total": 100.0}, config=cfg)
        assert r.metrics["coverage_leaf_score"].representative() == pytest.approx(0.5)


class TestRulesOnly:
    def test_all_pass(self):
        cfg = EvalConfig(
            metrics=[
                RulePassRate(
                    rules=[
                        Rule("$.status").eq("paid"),
                        Rule("$.total").eq("$.subtotal + $.tax"),
                    ]
                )
            ]
        )
        doc = {"status": "paid", "total": 110.0, "subtotal": 100.0, "tax": 10.0}
        r = evaluate(doc, config=cfg)
        assert r.metrics["rule_pass_rate"].representative() == 1.0
        assert len(r.metrics["rule_pass_rate"].extra_values("rule_results")) == 2

    def test_partial(self):
        cfg = EvalConfig(
            metrics=[
                RulePassRate(
                    rules=[
                        Rule("$.status").eq("paid"),
                        Rule("$.total").gt(0),
                    ]
                )
            ]
        )
        r = evaluate({"status": "draft", "total": 50.0}, config=cfg)
        assert r.metrics["rule_pass_rate"].representative() == pytest.approx(0.5)


class TestCombined:
    def test_schema_and_rules_as_score(self):
        cfg = EvalConfig(
            metrics=[SchemaValidity(Invoice)],
            key_metric=RulePassRate(rules=[Rule("$.total").gt(0)]),
        )
        r = evaluate({"id": "1", "total": 100.0, "status": "paid"}, config=cfg)
        assert r.metrics["schema_validity"].representative() == 1.0
        assert r.score == 1.0
        assert r.score_label == "rule_pass_rate"


class TestNoExpectedNoMetrics:
    @pytest.mark.xfail(
        reason="no-expected → None is deferred (NullPolicy / no-ground-truth semantics); "
        "the default ObjectAccuracy currently scores a value-less doc vacuously 1.0",
        strict=True,
    )
    def test_field_scores_have_no_value_score(self):
        # no expected, no metrics → no ground truth → score should be None
        r = evaluate({"x": 1}, config=EvalConfig())
        assert r.score is None
