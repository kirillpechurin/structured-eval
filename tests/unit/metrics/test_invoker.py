"""MetricInvoker — the one way to run a metric, in either input mode.

Nothing calls `compute` / `compute_<kind>` / `score` directly. Two modes:
`on_node` when a node exists (the engine), `on_values` when only raw values
do (array alignment, before any node is built). A `GenericMetric` dispatches
per kind in both; anything else uses `compute` / `score`.
"""

from collections.abc import Callable
from typing import Any

import pytest

from structured_eval.metrics.base import AnyNodeMetric, GenericMetric
from structured_eval.metrics.invoker import MetricInvoker, _kind_of
from structured_eval.models import ArrayNode, EvalContext, EvalNode, ObjectNode
from structured_eval.models.nodes.scalar import ScalarNode

pytestmark = pytest.mark.unit


class Plain(AnyNodeMetric):
    """A non-generic metric: one `compute`, one `score`."""

    name = "plain_probe"

    def compute(self, node: EvalNode) -> float:
        return 0.25

    def score(self, actual: Any, expected: Any) -> float:
        return 0.75


class PerKind(GenericMetric):
    """A generic metric implementing every kind, in both modes."""

    name = "per_kind_probe"

    def compute_scalar(self, node: ScalarNode) -> float:
        return 0.1

    def compute_object(self, node: ObjectNode) -> float:
        return 0.2

    def compute_array(self, node: ArrayNode) -> float:
        return 0.3

    def score_scalar(self, actual: Any, expected: Any) -> float:
        return 0.4

    def score_object(self, actual: Any, expected: Any) -> float:
        return 0.5

    def score_array(self, actual: Any, expected: Any) -> float:
        return 0.6


class ObjectsOnly(GenericMetric):
    """A generic metric that only knows about objects."""

    name = "objects_only_probe"

    def compute_object(self, node: ObjectNode) -> float:
        return 1.0


class Splitting(AnyNodeMetric):
    """Reports several sub-scores — no single number to narrow to."""

    name = "splitting_probe"

    def compute(self, node: EvalNode) -> dict[str, float]:
        return {"left": 0.0, "right": 1.0}

    def score(self, actual: Any, expected: Any) -> dict[str, float]:
        return {"left": 0.0, "right": 1.0}


def _nodes(context: EvalContext) -> dict[str, EvalNode]:
    """One node of each type, all sharing the context."""
    return {
        "scalar": ScalarNode(path="total", context=context),
        "object": ObjectNode(path="vendor", context=context),
        "array": ArrayNode(path="lines", context=context),
    }


# ── which value shape maps to which node kind ────────────────────────────────


@pytest.mark.parametrize(
    ("actual", "expected", "kind"),
    [
        ({"a": 1}, {"a": 1}, ObjectNode),
        ([1, 2], [1, 2], ArrayNode),
        ("x", "y", ScalarNode),
        (None, {"a": 1}, ObjectNode),
        ({"a": 1}, None, ObjectNode),  # expected absent → the actual decides
        (None, None, ScalarNode),
    ],
    ids=["dicts", "lists", "scalars", "expected-only", "actual-only", "both-null"],
)
def test_kind_is_inferred_the_way_the_tree_would_build_it(
    actual: Any, expected: Any, kind: type
) -> None:
    # `on_values` has no node to look at, so it infers the kind from the values
    # — mirroring TreeBuilder, which prefers `expected` as the reference shape.
    assert _kind_of(actual, expected) is kind


# ── on_node ──────────────────────────────────────────────────────────────────


def test_a_plain_metric_computes_whatever_the_node_is(
    context_factory: Callable[..., EvalContext],
) -> None:
    nodes = _nodes(context_factory({"total": 1}))
    for node in nodes.values():
        assert MetricInvoker(Plain()).on_node(node) == 0.25


@pytest.mark.parametrize(
    ("kind", "value"),
    [("scalar", 0.1), ("object", 0.2), ("array", 0.3)],
    ids=["scalar", "object", "array"],
)
def test_a_generic_metric_dispatches_on_the_node_kind(
    context_factory: Callable[..., EvalContext], kind: str, value: float
) -> None:
    node = _nodes(context_factory({"total": 1}))[kind]
    assert MetricInvoker(PerKind()).on_node(node) == value


@pytest.mark.parametrize("kind", ["scalar", "array"], ids=["scalar", "array"])
def test_a_generic_metric_is_silent_on_a_kind_it_does_not_implement(
    context_factory: Callable[..., EvalContext], kind: str
) -> None:
    # `TreeBuilder` already keeps it off those nodes; this is the same answer
    # for anyone who invokes it anyway — nothing, rather than an error.
    node = _nodes(context_factory({"total": 1}))[kind]
    assert MetricInvoker(ObjectsOnly()).on_node(node) is None


# ── on_values ────────────────────────────────────────────────────────────────


def test_a_plain_metric_compares_values_with_score() -> None:
    assert MetricInvoker(Plain()).on_values("a", "b") == 0.75


@pytest.mark.parametrize(
    ("actual", "expected", "value"),
    [("a", "b", 0.4), ({"a": 1}, {"a": 2}, 0.5), ([1], [2], 0.6)],
    ids=["scalar", "object", "array"],
)
def test_a_generic_metric_dispatches_on_the_inferred_kind(
    actual: Any, expected: Any, value: float
) -> None:
    assert MetricInvoker(PerKind()).on_values(actual, expected) == value


def test_a_generic_metric_is_silent_on_values_it_has_no_score_for() -> None:
    assert MetricInvoker(ObjectsOnly()).on_values("a", "b") is None


# ── narrowing to a single number ─────────────────────────────────────────────


def test_scalar_modes_narrow_the_result(
    context_factory: Callable[..., EvalContext],
) -> None:
    node = _nodes(context_factory({"total": 1}))["scalar"]
    assert MetricInvoker(Plain()).scalar_on_node(node) == 0.25
    assert MetricInvoker(Plain()).scalar_on_values("a", "b") == 0.75


def test_narrowing_a_multi_score_result_is_refused(
    context_factory: Callable[..., EvalContext],
) -> None:
    # Callers that need one verdict (array alignment) say so by asking for a
    # scalar; a metric reporting several sub-scores cannot answer them.
    node = _nodes(context_factory({"total": 1}))["scalar"]
    with pytest.raises(AssertionError, match="must yield a scalar at total"):
        MetricInvoker(Splitting()).scalar_on_node(node)
    with pytest.raises(AssertionError, match="must yield a scalar at <values>"):
        MetricInvoker(Splitting()).scalar_on_values("a", "b")
