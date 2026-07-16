"""End-to-end tests through the Evaluator (parse → tree → metrics → report).

Exercises the full three-phase pipeline on realistic documents, the side
channels (schema_errors / rule_results / hallucinated_fields), warnings, the
key-metric → report.score wiring, and parse-error handling.
"""

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import BaseModel

from structured_eval.metrics import (
    CoverageLeafScore,
    ExactMatch,
    FieldFaithfulness,
    ObjectAccuracy,
    ObjectF1,
    OverallLeafScore,
    RulePassRate,
    SchemaValidity,
    TokenF1,
)
from structured_eval.metrics.rule_pass_rate.dsl import Rule
from structured_eval.models import (
    ArrayFieldConfig,
    ArrayStrategy,
    EvalConfig,
    EvalReport,
    ExtraKeysPolicy,
    FieldConfig,
    ObjectFieldConfig,
    WarningType,
)

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
    from structured_eval.metrics import ArrayF1

    doc = {"vendor": {"name": "Acme"}, "lines": [1, 2, 3]}
    cfg = EvalConfig(metrics=[ObjectF1(), ArrayF1()])
    r = evaluate_one(doc, doc, cfg)
    assert r.field_scores["vendor.name"].score == 1.0
    assert "lines" in r.array_matches


# ── arrays of primitives (#3) ────────────────────────────────────────────────
#
# TreeBuilder.node dispatches on the *value*, so an array element that is a
# primitive becomes a ScalarNode and is scored like any other leaf — no wrapping
# in an object required.


@pytest.mark.parametrize(
    ("actual", "expected", "scores"),
    [
        ([1, 2, 3], [1, 2, 9], [1.0, 1.0, 0.0]),
        (["ai", "ml", "nlp"], ["ai", "ml", "cv"], [1.0, 1.0, 0.0]),
        ([True, False, True], [True, True, True], [1.0, 0.0, 1.0]),
    ],
    ids=["numeric", "string", "boolean"],
)
def test_primitive_elements_scored_individually(
    evaluate_one: Callable[..., EvalReport],
    actual: list[Any],
    expected: list[Any],
    scores: list[float],
) -> None:
    # BY_INDEX (the default) pairs the i-th with the i-th, so every element is
    # reachable at its own path rather than collapsing into one array verdict.
    r = evaluate_one({"tags": actual}, {"tags": expected})
    assert [r.field_scores[f"tags[{i}]"].score for i in range(len(actual))] == scores


def test_primitive_array_matches_report_missed_and_spurious(
    evaluate_one: Callable[..., EvalReport],
) -> None:
    # Matching scalars by value (BY_KEY with no key) gives set semantics:
    # ai + python pair up, ml (1) and nlp (3) are missed, llm (1) is spurious.
    cfg = EvalConfig(fields={"tags": ArrayFieldConfig(strategy=ArrayStrategy.BY_KEY)})
    r = evaluate_one(
        {"tags": ["ai", "llm", "python"]}, {"tags": ["ai", "ml", "python", "nlp"]}, cfg
    )

    m = r.array_matches["tags"]
    assert m.matched == [(0, 0), (2, 2)]
    assert m.missed == [1, 3]
    assert m.spurious == [1]
    # Only matched pairs become element nodes — an unmatched element has no
    # counterpart to score against, so array_matches is where it is reported.
    assert sorted(p for p in r.field_scores if p.startswith("tags[")) == [
        "tags[0]",
        "tags[2]",
    ]


# ── AnyNodeMetric (cascades uniformly onto every node) ───────────────────────


def test_any_node_metric_cascades_onto_every_node(
    evaluate_one: Callable[..., EvalReport],
) -> None:
    from structured_eval.metrics.base import AnyNodeMetric
    from structured_eval.models.nodes.base import EvalNode

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
    from structured_eval.models.nodes.base import EvalNode

    class Half(AnyNodeMetric):
        name = "half"

        def compute(self, node: EvalNode) -> float:
            return 0.5

    r = evaluate_one(
        {"a": 1}, {"a": 1}, EvalConfig(metrics=[Half()], key_metric="half")
    )
    assert r.score_label == "half"
    assert r.score == 0.5


# ── per-node key_metric override on object / array (#49) ─────────────────────
# ObjectFieldConfig / ArrayFieldConfig can pick a node's representative (roll-up)
# metric, the same override FieldConfig offers for scalar leaves. Each case
# carries *two* metrics whose values diverge, so the override is observable: the
# representative equals the chosen metric, not the default MeanScore of both.


def test_object_key_metric_override(evaluate_one: Callable[..., EvalReport]) -> None:
    # vendor: x,y correct, z missing → object_f1 (0.8) ≠ object_accuracy (0.667).
    actual = {"vendor": {"x": 1, "y": 2}}
    expected = {"vendor": {"x": 1, "y": 2, "z": 3}}
    metrics = [ObjectF1(), ObjectAccuracy()]

    default = evaluate_one(
        actual,
        expected,
        EvalConfig(fields={"vendor": ObjectFieldConfig(metrics=metrics)}),
    )
    override = evaluate_one(
        actual,
        expected,
        EvalConfig(
            fields={"vendor": ObjectFieldConfig(metrics=metrics, key_metric=ObjectF1())}
        ),
    )

    f1 = float(override.metrics["object_f1"].by_path["vendor"])
    acc = float(override.metrics["object_accuracy"].by_path["vendor"])
    assert f1 != acc  # the two metrics genuinely diverge on this node
    # default: representative is MeanScore of the node's two metrics
    assert default.field_scores["vendor"].score == pytest.approx((f1 + acc) / 2)
    # override: representative is object_f1 exactly — not the mean
    assert override.field_scores["vendor"].score == pytest.approx(f1)
    assert override.field_scores["vendor"].score != default.field_scores["vendor"].score


def test_array_key_metric_override(evaluate_one: Callable[..., EvalReport]) -> None:
    from structured_eval.metrics import ArrayAccuracy, ArrayF1

    # lines: 2 correct, 1 wrong, 1 missing → array_f1 (0.571) ≠ array_accuracy (0.667).
    actual = {"lines": [1, 2, 3, 4]}
    expected = {"lines": [1, 2, 9]}
    metrics = [ArrayF1(), ArrayAccuracy()]

    default = evaluate_one(
        actual,
        expected,
        EvalConfig(fields={"lines": ArrayFieldConfig(metrics=metrics)}),
    )
    override = evaluate_one(
        actual,
        expected,
        EvalConfig(
            fields={"lines": ArrayFieldConfig(metrics=metrics, key_metric=ArrayF1())}
        ),
    )

    f1 = float(override.metrics["array_f1"].by_path["lines"])
    acc = float(override.metrics["array_accuracy"].by_path["lines"])
    assert f1 != acc
    assert default.field_scores["lines"].score == pytest.approx((f1 + acc) / 2)
    assert override.field_scores["lines"].score == pytest.approx(f1)
    assert override.field_scores["lines"].score != default.field_scores["lines"].score


def test_object_key_metric_rolls_up_into_parent(
    evaluate_one: Callable[..., EvalReport],
) -> None:
    # a soft root metric reads each child's representative, so overriding vendor's
    # representative changes what rolls up into the root — the point of the feature.
    actual = {"vendor": {"x": 1, "y": 2}, "id": "K"}
    expected = {"vendor": {"x": 1, "y": 2, "z": 3}, "id": "K"}
    metrics = [ObjectF1(), ObjectAccuracy()]
    root = EvalConfig(metrics=[ObjectAccuracy()])

    default = evaluate_one(
        actual,
        expected,
        root.model_copy(
            update={"fields": {"vendor": ObjectFieldConfig(metrics=metrics)}}
        ),
    )
    override = evaluate_one(
        actual,
        expected,
        root.model_copy(
            update={
                "fields": {
                    "vendor": ObjectFieldConfig(metrics=metrics, key_metric=ObjectF1())
                }
            }
        ),
    )

    # vendor's representative differs between the two configs …
    assert override.field_scores["vendor"].score != default.field_scores["vendor"].score
    # … and that difference propagates into the root's soft accuracy roll-up.
    assert (
        override.metrics["object_accuracy"].representative()
        != default.metrics["object_accuracy"].representative()
    )


def test_object_array_config_without_key_metric_unchanged(
    evaluate_one: Callable[..., EvalReport],
) -> None:
    # an object/array config with no key_metric leaves the representative as-is:
    # identical to evaluating with no per-field config at all.
    actual = {"vendor": {"name": "Acme", "city": "NY"}, "lines": [1, 2, 3]}
    expected = {"vendor": {"name": "Acme", "city": "LA"}, "lines": [1, 2, 9]}
    baseline = evaluate_one(actual, expected)
    cfg = EvalConfig(
        fields={"vendor": ObjectFieldConfig(), "lines": ArrayFieldConfig()}
    )
    r = evaluate_one(actual, expected, cfg)
    assert r.field_scores["vendor"].score == baseline.field_scores["vendor"].score
    assert r.field_scores["lines"].score == baseline.field_scores["lines"].score


# ── incompatible metric assignment fails fast ────────────────────────────────
# An explicit per-node metric that cannot score the node's type is a config
# mistake — deterministic, input-independent — so TreeBuilder raises at build
# time instead of silently dropping it. Globals cascaded from EvalConfig stay
# exempt: cascading-by-type is intended, so a global that does not fit is filtered.

# doc is a perfect self-match: any raise is about the config, never the data.
_DOC = {"vendor": {"name": "Acme"}, "total": 100.0}


@pytest.mark.parametrize(
    ("config", "metric_name", "path", "node_type"),
    [
        # scalar-only metric on an object field
        (
            EvalConfig(fields={"vendor": ObjectFieldConfig(metrics=[ExactMatch()])}),
            "exact_match",
            "vendor",
            "ObjectNode",
        ),
        # object metric on a scalar field
        (
            EvalConfig(fields={"total": FieldConfig(metrics=[ObjectAccuracy()])}),
            "object_accuracy",
            "total",
            "ScalarNode",
        ),
        # an explicit per-node key_metric is checked the same way — on a scalar
        # FieldConfig and (since #49) on object/array configs too
        (
            EvalConfig(fields={"total": FieldConfig(key_metric=ObjectAccuracy())}),
            "object_accuracy",
            "total",
            "ScalarNode",
        ),
        (
            EvalConfig(fields={"vendor": ObjectFieldConfig(key_metric=ExactMatch())}),
            "exact_match",
            "vendor",
            "ObjectNode",
        ),
    ],
    ids=[
        "scalar-on-object",
        "object-on-scalar",
        "incompatible-key-metric-scalar",
        "incompatible-key-metric-object",
    ],
)
def test_incompatible_metric_raises(
    evaluate_one: Callable[..., EvalReport],
    config: EvalConfig,
    metric_name: str,
    path: str,
    node_type: str,
) -> None:
    with pytest.raises(ValueError, match="cannot score") as exc:
        evaluate_one(_DOC, _DOC, config)
    message = str(exc.value)
    assert metric_name in message  # names the metric
    assert path in message  # names the field path
    assert node_type in message  # names the node type


def test_global_cascaded_metric_does_not_raise(
    evaluate_one: Callable[..., EvalReport],
) -> None:
    # ObjectAccuracy cascades globally: it fits the objects and is silently
    # filtered from the scalar ``total`` node — never raised.
    r = evaluate_one(_DOC, _DOC, EvalConfig(metrics=[ObjectAccuracy()]))
    assert "object_accuracy" in r.field_scores["vendor"].metrics
    assert "object_accuracy" not in r.field_scores["total"].metrics


def test_global_key_metric_does_not_raise(
    evaluate_one: Callable[..., EvalReport],
) -> None:
    # The global key_metric is distributable: applied where it fits, else ignored.
    r = evaluate_one(_DOC, _DOC, EvalConfig(key_metric=ObjectAccuracy()))
    assert isinstance(r, EvalReport)
