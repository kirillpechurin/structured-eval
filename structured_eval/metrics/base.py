from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from structured_eval.model.nodes.array_node import ArrayNode
from structured_eval.model.nodes.base import EvalNode
from structured_eval.model.nodes.object_node import ObjectNode
from structured_eval.model.nodes.scalar import ScalarNode

# Name → metric class. Populated automatically as BaseMetric subclasses are
# declared; used by EvalConfig.from_yaml() to resolve string names (Stage 10).
_METRIC_REGISTRY: dict[str, type] = {}


class BaseMetric(ABC):
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

    def compute(self, node: NodeT) -> float | dict[str, float] | None:
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
    def compute(self, node: ObjectNode) -> float | dict[str, float] | None: ...


class ArrayMetric(Metric[ArrayNode]):
    """Applies to each ArrayNode."""

    @abstractmethod
    def compute(self, node: ArrayNode) -> float | dict[str, float] | None: ...


class RootMetric(Metric[EvalNode]):
    """Applies only to the root node (path == "$"); receives any EvalNode."""

    @abstractmethod
    def compute(self, node: EvalNode) -> float | dict[str, float] | None: ...


class GenericMetric(BaseMetric):
    """Metrics spanning several node types, outside the single-``compute`` shape.

    Override whichever of ``compute_scalar`` / ``compute_object`` /
    ``compute_array`` apply; ``MetricRunner`` dispatches by node type and
    ``TreeBuilder`` admits it onto a node only when the matching method exists.
    (Replaces the former ``NodeMetric``.)
    """


def get_metric_class(name: str) -> type:
    """Resolve a metric class by its ``name`` (e.g. ``"object_f1"``)."""
    if name not in _METRIC_REGISTRY:
        raise KeyError(f"Unknown metric: {name!r}. Known: {sorted(_METRIC_REGISTRY)}")
    return _METRIC_REGISTRY[name]


def resolve_metric(spec: Any) -> BaseMetric:
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
