"""Per-type default metrics: EvalConfig.default_{scalar,object,array}_metrics.

The default is the *fallback* layer, not a cascade: it lands only on nodes that
would otherwise carry no metric at all, replacing the hard-coded ExactMatch /
ObjectAccuracy / ArrayAccuracy. Anything more specific — a cascading
config.metrics, the node's own cfg.metrics, an explicit key_metric — wins.
"""

from collections.abc import Callable
from typing import Any

import pytest

from structured_eval.metrics import (
    ArrayAccuracy,
    ArrayF1,
    ExactMatch,
    Numeric,
    ObjectAccuracy,
    ObjectF1,
    TokenF1,
)
from structured_eval.models import (
    ArrayFieldConfig,
    EvalConfig,
    EvalReport,
    FieldConfig,
    ObjectFieldConfig,
)

pytestmark = pytest.mark.engine

# One document exercising all three node types at once, each wrong in a way that
# separates the built-in default from the configured one:
#   total  0.5% off → ExactMatch 0.0, Numeric(0.1) 1.0
#   vendor a right, b wrong, c missing → ObjectAccuracy 0.5, ObjectF1 2*1/(2+3)
#   lines  one short → ArrayAccuracy 2/3, ArrayF1 2*2/(2+3)
ACTUAL: dict[str, Any] = {"total": 100.5, "vendor": {"a": 1, "b": 9}, "lines": [1, 2]}
EXPECTED: dict[str, Any] = {
    "total": 100.0,
    "vendor": {"a": 1, "b": 2, "c": 3},
    "lines": [1, 2, 3],
}


# ── the default applies where nothing more specific does ─────────────────────


@pytest.mark.parametrize(
    ("config", "path", "metric", "score"),
    [
        (
            EvalConfig(default_scalar_metrics=[Numeric(tolerance=0.1)]),
            "total",
            "numeric",
            1.0,
        ),
        (
            EvalConfig(default_object_metrics=[ObjectF1()]),
            "vendor",
            "object_f1",
            0.4,
        ),
        (
            EvalConfig(default_array_metrics=[ArrayF1()]),
            "lines",
            "array_f1",
            0.8,
        ),
    ],
    ids=["scalar", "object", "array"],
)
def test_default_replaces_the_built_in(
    evaluate_one: Callable[..., EvalReport],
    config: EvalConfig,
    path: str,
    metric: str,
    score: float,
) -> None:
    report = evaluate_one(ACTUAL, EXPECTED, config)

    field = report.field_scores[path]
    assert set(field.metrics) == {metric, "mean_score"}  # the built-in is gone
    assert field.score == pytest.approx(score)


def test_defaults_reach_every_node_of_their_type(
    evaluate_one: Callable[..., EvalReport],
) -> None:
    """The fallback is decided per node, so nesting depth changes nothing."""
    config = EvalConfig(default_scalar_metrics=[Numeric(tolerance=0.1)])
    report = evaluate_one(
        {"vendor": {"totals": [{"total": 100.5}]}},
        {"vendor": {"totals": [{"total": 100.0}]}},
        config,
    )
    assert report.field_scores["vendor.totals[0].total"].score == 1.0


def test_one_default_instance_serves_every_node(
    tree_factory: Callable[..., Any],
) -> None:
    """The default is resolved once and shared — metrics carry no per-node state."""
    config = EvalConfig(default_scalar_metrics=[Numeric(tolerance=0.1)])
    root = tree_factory({"a": 100.5, "b": 100.5}, {"a": 100.0, "b": 200.0}, config)

    assert root.children["a"].metric_results["numeric"] == 1.0
    assert root.children["b"].metric_results["numeric"] == 0.0


# ── anything more specific wins ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("config", "path", "metric"),
    [
        (
            EvalConfig(
                default_scalar_metrics=[Numeric(tolerance=0.1)],
                fields={"total": FieldConfig(metrics=[ExactMatch()])},
            ),
            "total",
            "exact_match",
        ),
        (
            EvalConfig(
                default_object_metrics=[ObjectF1()],
                fields={"vendor": ObjectFieldConfig(metrics=[ObjectAccuracy()])},
            ),
            "vendor",
            "object_accuracy",
        ),
        (
            EvalConfig(
                default_array_metrics=[ArrayF1()],
                fields={"lines": ArrayFieldConfig(metrics=[ArrayAccuracy()])},
            ),
            "lines",
            "array_accuracy",
        ),
    ],
    ids=["scalar", "object", "array"],
)
def test_node_metrics_win_over_the_default(
    evaluate_one: Callable[..., EvalReport],
    config: EvalConfig,
    path: str,
    metric: str,
) -> None:
    report = evaluate_one(ACTUAL, EXPECTED, config)

    # the node is non-empty already, so the default is never reached
    assert set(report.field_scores[path].metrics) == {metric, "mean_score"}


def test_cascading_metrics_win_over_the_default(
    evaluate_one: Callable[..., EvalReport],
) -> None:
    """A global that fits the node leaves it non-empty, so the default never fires."""
    config = EvalConfig(
        metrics=[TokenF1()], default_scalar_metrics=[Numeric(tolerance=0.1)]
    )
    report = evaluate_one({"name": "acme corp"}, {"name": "acme"}, config)

    assert set(report.field_scores["name"].metrics) == {"token_f1", "mean_score"}


def test_explicit_key_metric_wins_over_the_default(
    evaluate_one: Callable[..., EvalReport],
) -> None:
    """key_metric picks the representative; the default still supplies the metric."""
    config = EvalConfig(
        default_scalar_metrics=[Numeric(tolerance=0.1)],
        fields={"total": FieldConfig(key_metric=ExactMatch())},
    )
    report = evaluate_one(ACTUAL, EXPECTED, config)

    field = report.field_scores["total"]
    assert field.score == 0.0  # representative = exact_match, not numeric
    assert field.metrics["numeric"] == 1.0  # the default ran all the same


# ── omitting the option changes nothing ──────────────────────────────────────


def test_omitting_defaults_reproduces_the_built_ins(
    evaluate_one: Callable[..., EvalReport],
) -> None:
    baseline = evaluate_one(ACTUAL, EXPECTED)
    explicit = evaluate_one(
        ACTUAL,
        EXPECTED,
        EvalConfig(
            default_scalar_metrics=[ExactMatch()],
            default_object_metrics=[ObjectAccuracy()],
            default_array_metrics=[ArrayAccuracy()],
        ),
    )
    assert explicit.score == baseline.score
    assert set(explicit.metrics) == set(baseline.metrics)


# ── configuration mistakes fail fast ─────────────────────────────────────────


@pytest.mark.parametrize(
    ("config", "metric", "node_type"),
    [
        (
            EvalConfig(default_scalar_metrics=[ObjectAccuracy()]),
            "object_accuracy",
            "ScalarNode",
        ),
        (
            EvalConfig(default_object_metrics=[ExactMatch()]),
            "exact_match",
            "ObjectNode",
        ),
        (EvalConfig(default_array_metrics=[ExactMatch()]), "exact_match", "ArrayNode"),
    ],
    ids=["object-metric-as-scalar-default", "scalar-as-object", "scalar-as-array"],
)
def test_default_that_cannot_score_its_node_type_raises(
    evaluate_one: Callable[..., EvalReport],
    config: EvalConfig,
    metric: str,
    node_type: str,
) -> None:
    # a perfect self-match: any raise is about the config, never the data.
    doc = {"vendor": {"total": 100.0}, "lines": [1, 2]}
    with pytest.raises(ValueError, match="cannot score") as exc:
        evaluate_one(doc, doc, config)
    message = str(exc.value)
    assert metric in message  # names the metric
    assert node_type in message  # names the node type


def test_two_defaults_sharing_a_name_raise(
    evaluate_one: Callable[..., EvalReport],
) -> None:
    """Same rule as any other layer: one name, one metric per node."""
    config = EvalConfig(
        default_scalar_metrics=[Numeric(tolerance=0.001), Numeric(tolerance=0.1)]
    )
    with pytest.raises(ValueError, match="assigned twice"):
        evaluate_one(ACTUAL, EXPECTED, config)


@pytest.mark.parametrize(
    "field",
    ["default_scalar_metrics", "default_object_metrics", "default_array_metrics"],
)
def test_an_empty_default_list_is_rejected(field: str) -> None:
    """[] would leave nodes with no metric at all — MeanScore would read 0.0."""
    with pytest.raises(ValueError, match="at least one metric"):
        EvalConfig.model_validate({field: []})
