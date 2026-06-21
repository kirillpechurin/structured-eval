"""End-to-end tests through the Evaluator (parse → tree → metrics → report).

Exercises the full three-phase pipeline on realistic documents, the side
channels (schema_errors / rule_results / hallucinated_fields), warnings, the
key-metric → report.score wiring, and parse-error handling.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from structured_eval import (
    Coverage,
    EvalConfig,
    ExtraKeysPolicy,
    Faithfulness,
    FieldConfig,
    ObjectF1,
    OverallScore,
    RulePassRate,
    SchemaValidity,
    TokenF1,
)
from structured_eval.metrics.rule_pass_rate.dsl import Rule

pytestmark = pytest.mark.engine


class Invoice(BaseModel):
    id: str
    total: float
    status: str


class TestBasicEval:
    def test_perfect_object(self, evaluate_one, invoice_pair):
        actual, expected = invoice_pair
        r = evaluate_one(actual, expected, EvalConfig(metrics=[ObjectF1()]))
        # total differs (99 vs 100) → not perfect
        assert r.metrics["object_f1"].representative() == pytest.approx(0.75)

    def test_field_scores_populated(self, evaluate_one, invoice_pair):
        actual, expected = invoice_pair
        r = evaluate_one(actual, expected)
        assert r.field_scores["total"].score == 0.0
        assert r.field_scores["id"].score == 1.0

    def test_key_metric_becomes_score(self, evaluate_one, invoice_pair):
        actual, expected = invoice_pair
        cfg = EvalConfig(key_metric=ObjectF1())
        r = evaluate_one(actual, expected, cfg)
        assert r.score_label == "object_f1"
        assert r.score == r.metrics["object_f1"].representative()


class TestParseErrors:
    def test_invalid_json_actual(self, evaluate_one):
        r = evaluate_one("{bad json", {"a": 1})
        assert r.parse_error
        assert r.parse_error_message
        assert r.metrics == {}

    def test_valid_json_string_parsed(self, evaluate_one):
        r = evaluate_one('{"a": 1}', {"a": 1}, EvalConfig(metrics=[ObjectF1()]))
        assert not r.parse_error
        assert r.metrics["object_f1"].representative() == 1.0

    def test_yaml_fallback(self, evaluate_one):
        r = evaluate_one("a: 1\nb: 2", {"a": 1, "b": 2}, EvalConfig(metrics=[ObjectF1()]))
        assert not r.parse_error
        assert r.metrics["object_f1"].representative() == 1.0


class TestSideChannels:
    def test_schema_errors_surface(self, evaluate_one):
        cfg = EvalConfig(metrics=[SchemaValidity(Invoice)])
        r = evaluate_one({"id": "1"}, None, cfg)
        assert r.metrics["schema_validity"].representative() == 0.0
        assert r.metrics["schema_validity"].extra_values("schema_errors")

    def test_rule_results_surface(self, evaluate_one):
        cfg = EvalConfig(
            metrics=[
                RulePassRate(
                    rules=[
                        Rule("$.total").eq("$.subtotal + $.tax"),
                        Rule("$.status").eq("paid"),
                    ]
                )
            ]
        )
        doc = {"total": 110.0, "subtotal": 100.0, "tax": 10.0, "status": "paid"}
        r = evaluate_one(doc, None, cfg)
        assert r.metrics["rule_pass_rate"].representative() == 1.0
        assert len(r.metrics["rule_pass_rate"].extra_values("rule_results")) == 2

    def test_hallucinations_surface(self, evaluate_one, invoice_source):
        cfg = EvalConfig(metrics=[Faithfulness()])
        r = evaluate_one({"vendor": "Globex"}, None, cfg, source=invoice_source)
        assert r.metrics["faithfulness"].representative() == 0.0
        assert r.metrics["faithfulness"].extra_values("hallucinated_fields") == ["vendor"]

    def test_faithfulness_requires_source(self, evaluate_one):
        cfg = EvalConfig(metrics=[Faithfulness()])
        with pytest.raises(ValueError, match="source"):
            evaluate_one({"vendor": "Globex"}, None, cfg)


class TestWarnings:
    def test_extra_key_warning(self, evaluate_one):
        r = evaluate_one({"a": 1, "extra": 2}, {"a": 1}, EvalConfig(metrics=[ObjectF1()]))
        assert any("EXTRA_KEY" in w for w in r.warnings)

    def test_missing_field_warning(self, evaluate_one):
        r = evaluate_one({"a": 1}, {"a": 1, "b": 2}, EvalConfig(metrics=[ObjectF1()]))
        assert any("MISSING_FIELD" in w for w in r.warnings)

    def test_extra_key_penalized(self, evaluate_one):
        cfg = EvalConfig(metrics=[ObjectF1()], extra_keys=ExtraKeysPolicy.PENALIZE)
        r = evaluate_one({"a": 1, "extra": 2}, {"a": 1}, cfg)
        assert r.metrics["object_f1"].representative() < 1.0


class TestMultiMetric:
    def test_several_metrics_one_pass(self, evaluate_one, invoice_pair):
        actual, expected = invoice_pair
        cfg = EvalConfig(metrics=[ObjectF1(), Coverage(), OverallScore()])
        r = evaluate_one(actual, expected, cfg)
        assert {"object_f1", "coverage", "overall_score"} <= set(r.metrics)

    def test_per_field_metrics(self, evaluate_one):
        cfg = EvalConfig(
            fields={"name": FieldConfig(metrics=[TokenF1()], key_metric=TokenF1())},
            metrics=[ObjectF1()],
        )
        r = evaluate_one({"name": "the quick fox"}, {"name": "the quick brown fox"}, cfg)
        assert 0.0 < r.field_scores["name"].metrics["token_f1"] < 1.0

    def test_per_node_metric_at_depth(self, evaluate_one):
        # ObjectFieldConfig.metrics now applies to a nested object (recursive,
        # each node owns its metrics) — not only the global/root list.
        from structured_eval import ObjectFieldConfig

        cfg = EvalConfig(fields={"inner": ObjectFieldConfig(metrics=[ObjectF1()])})
        r = evaluate_one({"inner": {"a": 1, "b": 2}}, {"inner": {"a": 1, "b": 99}}, cfg)
        assert r.field_scores["inner"].metrics["object_f1"] == 0.5
        # the nested metric is local: it is not forced onto the root object
        assert "object_f1" not in r.field_scores["$"].metrics

    def test_nested_representative_bubbles_into_parent(self, evaluate_one):
        # A nested object's representative (key_metric = MeanScore over its
        # ObjectF1) is what the parent's ObjectF1 aggregates — object-in-object
        # is counted exactly like a scalar, not silently dropped.
        cfg = EvalConfig(metrics=[ObjectF1()])
        r = evaluate_one({"u": {"a": 1, "b": 9}, "x": 1}, {"u": {"a": 1, "b": 2}, "x": 1}, cfg)
        assert r.field_scores["u"].score == 0.5  # nested object's representative
        # root: x is a TP (1.0), u is not (0.5 < 1.0) → f1 = 0.5
        assert r.metrics["object_f1"].representative() == 0.5

    def test_default_key_metric_is_mean_of_node_metrics(self, evaluate_one):
        # report.score defaults to MeanScore: the mean of the root's own metrics.
        from structured_eval import ObjectAccuracy

        cfg = EvalConfig(metrics=[ObjectF1(), ObjectAccuracy()])
        r = evaluate_one({"a": 1, "b": 9}, {"a": 1, "b": 2}, cfg)
        assert r.score_label == "mean_score"
        assert r.score == pytest.approx(
            (r.metrics["object_f1"].representative() + r.metrics["object_accuracy"].representative())
            / 2
        )


class TestNested:
    def test_nested_object_and_array(self, evaluate_one):
        actual = {"vendor": {"name": "Acme"}, "lines": [1, 2, 3]}
        expected = {"vendor": {"name": "Acme"}, "lines": [1, 2, 3]}
        from structured_eval import ArrayF1

        cfg = EvalConfig(metrics=[ObjectF1(), ArrayF1()])
        r = evaluate_one(actual, expected, cfg)
        assert r.field_scores["vendor.name"].score == 1.0
        assert "lines" in r.array_matches
