from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from structured_eval.model.metric_result import MetricResult
from structured_eval.model.nodes.array_node import ArrayNode
from structured_eval.model.nodes.base import EvalNode
from structured_eval.model.nodes.object_node import ObjectNode
from structured_eval.model.nodes.scalar import ScalarNode

# What a metric's ``compute`` may return; ``MetricRunner._apply`` normalizes any
# of these to a ``MetricResult``. A bare value / dict of sub-scores, optionally
# paired with structured ``extra`` via a tuple, or a ready ``MetricResult``.
MetricOutput = (
    float
    | dict[str, float]
    | tuple[float | dict[str, float], dict[str, Any]]
    | MetricResult
    | None
)

# Name → metric class. Populated automatically as BaseMetric subclasses are
# declared; used by EvalConfig.from_yaml() to resolve string names (Stage 10).
_METRIC_REGISTRY: dict[str, type] = {}


class BaseMetric(ABC):  # noqa: B024 — registry root; subclasses define the interface
    """Registry root for every metric — no evaluation interface of its own.

    ``name`` is the key under which a scalar result lands in ``report.metrics``
    and ``FieldScore.metrics``. A metric that returns a ``dict`` instead writes
    each of its keys directly (the ``name`` is then only a registry handle).
    Declaring a subclass with a ``name`` registers it automatically.
    """

    name: str = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if n := getattr(cls, "name", None):
            _METRIC_REGISTRY[n] = cls


class Metric[NodeT: EvalNode](BaseMetric):
    """The unified metric interface: ``compute(node)`` + ``score(actual, expected)``.

    Every concrete metric is a ``Metric`` and therefore *has* a ``score`` — a
    pure value-level comparison ``(actual, expected) -> float | dict`` reused by
    array alignment. ``compute(node)`` is the node-level entry point; by default
    it delegates to ``score`` on the node's values, so a leaf comparison need
    only implement ``score``. Aggregating metrics override ``compute`` and leave
    ``score`` at its default (callers that require a scalar verdict check the
    result type where it matters). The type parameter ``NodeT`` pins the node
    type a subtype operates on (``ScalarNode`` for fields, ``ObjectNode`` …).
    """

    def compute(self, node: NodeT) -> MetricOutput:
        return self.score(node.actual, node.expected)

    def score(self, actual: Any, expected: Any) -> float | dict[str, float]:
        raise NotImplementedError


class FieldMetric(Metric[ScalarNode]):
    """A leaf comparison applied to each ScalarNode.

    Implements ``score(actual, expected)`` and relies on the inherited
    ``compute``; metrics that need node context (e.g. ``Presence``) override
    ``compute`` directly. Also the marker the engine dispatches on for scalars.
    """


class ObjectMetric(Metric[ObjectNode]):
    """Applies to each ObjectNode (root and nested)."""

    @abstractmethod
    def compute(self, node: ObjectNode) -> MetricOutput: ...


class ArrayMetric(Metric[ArrayNode]):
    """Applies to each ArrayNode."""

    @abstractmethod
    def compute(self, node: ArrayNode) -> MetricOutput: ...


class RootMetric(Metric[EvalNode]):
    """Applies only to the root node (path == "$"); receives any EvalNode."""

    @abstractmethod
    def compute(self, node: EvalNode) -> MetricOutput: ...


class AnyNodeMetric(Metric[EvalNode]):
    """Applies uniformly to *every* node — same ``compute`` regardless of kind.

    The node-agnostic branch of the hierarchy: unlike the typed metrics
    (``FieldMetric`` / ``ObjectMetric`` / ``ArrayMetric``) it is not pinned to
    one node type, and unlike ``GenericMetric`` it does not dispatch per kind —
    it runs one uniform computation on any ``EvalNode``. ``RootMetric`` is the
    sibling that is *also* ``Metric[EvalNode]`` but admitted only at the root;
    an ``AnyNodeMetric`` is admitted everywhere. ``MeanScore`` (the default
    representative) lives here, and a custom uniform metric can be cascaded via
    ``config.metrics`` or chosen as a ``key_metric``.
    """

    @abstractmethod
    def compute(self, node: EvalNode) -> MetricOutput: ...


class GenericMetric(BaseMetric):
    """Metrics spanning several node types, outside the single-``compute`` shape.

    Override whichever per-kind methods apply: ``compute_scalar`` /
    ``compute_object`` / ``compute_array`` for node mode, and (optionally)
    ``score_scalar`` / ``score_object`` / ``score_array`` for value mode.
    ``MetricInvoker`` dispatches by kind; ``TreeBuilder`` admits the metric onto
    a node only when the matching ``compute_<kind>`` exists. (Replaces the former
    ``NodeMetric``.)
    """


def get_metric_class(name: str) -> type:
    """Resolve a metric class by its ``name`` (e.g. ``"object_f1"``)."""
    if name not in _METRIC_REGISTRY:
        raise KeyError(f"Unknown metric: {name!r}. Known: {sorted(_METRIC_REGISTRY)}")
    return _METRIC_REGISTRY[name]


def resolve_metric(spec: str | BaseMetric) -> BaseMetric:
    """Coerce a metric spec to a ``BaseMetric`` instance.

    Accepts an instance as-is or a registered name string (instantiated with no
    args). The single resolver shared by the engine, array alignment, and the
    match-criterion helper. ``None`` is *not* handled here — callers supply
    their own default. Score-needing call sites narrow the result to ``Metric``.
    """
    if isinstance(spec, str):
        instance = get_metric_class(spec)()
        assert isinstance(instance, BaseMetric)
        return instance
    assert isinstance(spec, BaseMetric)
    return spec
