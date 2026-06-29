"""End-to-end tests through the Evaluator (parse → tree → metrics → report).

Exercises the full three-phase pipeline on realistic documents, the side
channels (schema_errors / rule_results / hallucinated_fields), warnings, the
key-metric → report.score wiring, and parse-error handling.
"""

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import BaseModel

from structured_eval import (
    CoverageLeafScore,
    EvalConfig,
    EvalReport,
    ExtraKeysPolicy,
    FieldConfig,
    FieldFaithfulness,
    ObjectF1,
    OverallLeafScore,
    RulePassRate,
    SchemaValidity,
    TokenF1,
    WarningType,
)
from structured_eval.metrics.rule_pass_rate.dsl import Rule

pytestmark = pytest.mark.engine


class Invoice(BaseModel):
    id: str
    total: float
    status: str


# ── basic evaluation ─────────────────────────────────────────────────────────


def test_perfect_object(
    evaluate_one: Callable[..., EvalReport],
    invoice_pair: tuple[dict[str, Any], dict[str, Any]],
    assert_metric: Callable[..., None],
) -> None:
    actual, expected = invoice_pair
    r = evaluate_one(actual, expected, EvalConfig(metrics=[ObjectF1()]))
    assert_metric(r, "object_f1", 0.75)  # total differs (99 vs 100)


def test_field_scores_populated(
    evaluate_one: Callable[..., EvalReport],
    invoice_pair: tuple[dict[str, Any], dict[str, Any]],
    assert_field: Callable[..., None],
) -> None:
    actual, expected = invoice_pair
    r = evaluate_one(actual, expected)
    assert_field(r, "total", 0.0)
    assert_field(r, "id", 1.0)


def test_key_metric_becomes_score(
    evaluate_one: Callable[..., EvalReport],
    invoice_pair: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    actual, expected = invoice_pair
    r = evaluate_one(actual, expected, EvalConfig(key_metric=ObjectF1()))
    assert r.score_label == "object_f1"
    assert r.score == r.metrics["object_f1"].representative()


# ── parsing ──────────────────────────────────────────────────────────────────


def test_invalid_json_actual(evaluate_one: Callable[..., EvalReport]) -> None:
    r = evaluate_one("{bad json", {"a": 1})
    assert r.parse_error
    assert r.parse_error_message
    assert r.metrics == {}


def test_valid_json_string_parsed(
    evaluate_one: Callable[..., EvalReport], assert_metric: Callable[..., None]
) -> None:
    r = evaluate_one('{"a": 1}', {"a": 1}, EvalConfig(metrics=[ObjectF1()]))
    assert not r.parse_error
    assert_metric(r, "object_f1", 1.0)


def test_yaml_fallback(
    evaluate_one: Callable[..., EvalReport], assert_metric: Callable[..., None]
) -> None:
    r = evaluate_one("a: 1\nb: 2", {"a": 1, "b": 2}, EvalConfig(metrics=[ObjectF1()]))
    assert not r.parse_error
    assert_metric(r, "object_f1", 1.0)


# ── side channels (.extra) ───────────────────────────────────────────────────


def test_schema_errors_surface(evaluate_one: Callable[..., EvalReport]) -> None:
    r = evaluate_one({"id": "1"}, None, EvalConfig(metrics=[SchemaValidity(Invoice)]))
    assert r.metrics["schema_validity"].representative() == 0.0
    assert r.metrics["schema_validity"].extra_values("schema_errors")


def test_rule_results_surface(evaluate_one: Callable[..., EvalReport]) -> None:
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


def test_hallucinations_surface(
    evaluate_one: Callable[..., EvalReport], invoice_source: str
) -> None:
    cfg = EvalConfig(metrics=[FieldFaithfulness()])
    r = evaluate_one({"vendor": "Globex"}, None, cfg, source=invoice_source)
    mc = r.metrics["field_faithfulness"]
    assert mc.mean() == 0.0
    # hallucinated fields are the leaves scoring 0.0
    assert [p for p, v in mc.by_path.items() if float(v) == 0.0] == ["vendor"]


def test_faithfulness_requires_source(evaluate_one: Callable[..., EvalReport]) -> None:
    with pytest.raises(ValueError, match="source"):
        evaluate_one(
            {"vendor": "Globex"}, None, EvalConfig(metrics=[FieldFaithfulness()])
        )


# ── warnings ─────────────────────────────────────────────────────────────────


def test_extra_key_warning(evaluate_one: Callable[..., EvalReport]) -> None:
    r = evaluate_one({"a": 1, "extra": 2}, {"a": 1}, EvalConfig(metrics=[ObjectF1()]))
    extra = [w for w in r.warnings if w.type == WarningType.EXTRA_KEY]
    assert [w.path for w in extra] == ["extra"]


def test_missing_field_warning(evaluate_one: Callable[..., EvalReport]) -> None:
    r = evaluate_one({"a": 1}, {"a": 1, "b": 2}, EvalConfig(metrics=[ObjectF1()]))
    missing = [w for w in r.warnings if w.type == WarningType.MISSING_FIELD]
    assert [w.path for w in missing] == ["b"]


def test_extra_key_penalized(evaluate_one: Callable[..., EvalReport]) -> None:
    cfg = EvalConfig(metrics=[ObjectF1()], extra_keys=ExtraKeysPolicy.PENALIZE)
    r = evaluate_one({"a": 1, "extra": 2}, {"a": 1}, cfg)
    assert r.metrics["object_f1"].representative() < 1.0


# ── multiple metrics / per-node ownership ────────────────────────────────────


def test_several_metrics_one_pass(
    evaluate_one: Callable[..., EvalReport],
    invoice_pair: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    actual, expected = invoice_pair
    cfg = EvalConfig(metrics=[ObjectF1(), CoverageLeafScore(), OverallLeafScore()])
    r = evaluate_one(actual, expected, cfg)
    assert {"object_f1", "coverage_leaf_score", "overall_leaf_score"} <= set(r.metrics)


def test_per_field_metrics(evaluate_one: Callable[..., EvalReport]) -> None:
    cfg = EvalConfig(
        fields={"name": FieldConfig(metrics=[TokenF1()], key_metric=TokenF1())},
        metrics=[ObjectF1()],
    )
    r = evaluate_one({"name": "the quick fox"}, {"name": "the quick brown fox"}, cfg)
    assert 0.0 < r.field_scores["name"].metrics["token_f1"] < 1.0


def test_per_node_metric_at_depth(evaluate_one: Callable[..., EvalReport]) -> None:
    # ObjectFieldConfig.metrics applies to a nested object (recursive, each node
    # owns its metrics) — not only the global/root list.
    from structured_eval import ObjectFieldConfig

    cfg = EvalConfig(fields={"inner": ObjectFieldConfig(metrics=[ObjectF1()])})
    r = evaluate_one({"inner": {"a": 1, "b": 2}}, {"inner": {"a": 1, "b": 99}}, cfg)
    assert r.field_scores["inner"].metrics["object_f1"] == 0.5
    # the nested metric is local: not forced onto the root object
    assert "object_f1" not in r.field_scores["$"].metrics


def test_nested_representative_bubbles_into_parent(
    evaluate_one: Callable[..., EvalReport],
) -> None:
    # A nested object's representative is what the parent's ObjectF1 aggregates —
    # object-in-object is counted exactly like a scalar, not silently dropped.
    cfg = EvalConfig(metrics=[ObjectF1()])
    r = evaluate_one(
        {"u": {"a": 1, "b": 9}, "x": 1}, {"u": {"a": 1, "b": 2}, "x": 1}, cfg
    )
    assert r.field_scores["u"].score == 0.5  # nested object's representative
    # root: x is a TP (1.0), u is not (0.5 < 1.0) → f1 = 0.5
    assert r.metrics["object_f1"].representative() == 0.5


def test_default_key_metric_is_mean_of_node_metrics(
    evaluate_one: Callable[..., EvalReport],
) -> None:
    # report.score defaults to MeanScore: the mean of the root's own metrics.
    from structured_eval import ObjectAccuracy

    cfg = EvalConfig(metrics=[ObjectF1(), ObjectAccuracy()])
    r = evaluate_one({"a": 1, "b": 9}, {"a": 1, "b": 2}, cfg)
    assert r.score_label == "mean_score"
    assert r.score == pytest.approx(
        (
            r.metrics["object_f1"].representative()
            + r.metrics["object_accuracy"].representative()
        )
        / 2
    )


# ── nested object + array ────────────────────────────────────────────────────


def test_nested_object_and_array(evaluate_one: Callable[..., EvalReport]) -> None:
    from structured_eval import ArrayF1

    doc = {"vendor": {"name": "Acme"}, "lines": [1, 2, 3]}
    cfg = EvalConfig(metrics=[ObjectF1(), ArrayF1()])
    r = evaluate_one(doc, doc, cfg)
    assert r.field_scores["vendor.name"].score == 1.0
    assert "lines" in r.array_matches


# ── AnyNodeMetric (cascades uniformly onto every node) ───────────────────────


def test_any_node_metric_cascades_onto_every_node(
    evaluate_one: Callable[..., EvalReport],
) -> None:
    from structured_eval.metrics.base import AnyNodeMetric
    from structured_eval.model.nodes.base import EvalNode

    class ConstDepth(AnyNodeMetric):
        name = "const_depth"

        def compute(self, node: EvalNode) -> float:
            return 0.42

    actual = {"vendor": {"name": "Acme"}, "lines": [1, 2], "total": 100}
    r = evaluate_one(actual, actual, EvalConfig(metrics=[ConstDepth()]))

    # one uniform metric, the same on root object, nested object, array, and
    # scalar leaf — no per-kind dispatch.
    for path in ("$", "vendor", "lines", "total", "vendor.name"):
        assert r.field_scores[path].metrics["const_depth"] == 0.42
    assert r.metrics["const_depth"].mean() == 0.42


def test_any_node_metric_usable_as_explicit_key_metric(
    evaluate_one: Callable[..., EvalReport],
) -> None:
    from structured_eval.metrics.base import AnyNodeMetric
    from structured_eval.model.nodes.base import EvalNode

    class Half(AnyNodeMetric):
        name = "half"

        def compute(self, node: EvalNode) -> float:
            return 0.5

    r = evaluate_one(
        {"a": 1}, {"a": 1}, EvalConfig(metrics=[Half()], key_metric="half")
    )
    assert r.score_label == "half"
    assert r.score == 0.5
