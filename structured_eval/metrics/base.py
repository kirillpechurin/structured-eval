from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from structured_eval.model.nodes.array_node import ArrayNode
from structured_eval.model.nodes.base import EvalNode
from structured_eval.model.nodes.object_node import ObjectNode
from structured_eval.model.nodes.scalar import ScalarNode

# Name → metric class. Populated automatically as Metric subclasses are
# declared; used by EvalConfig.from_yaml() to resolve string names (Stage 10).
_METRIC_REGISTRY: dict[str, type] = {}


class Metric(ABC):
    """Base for all metrics.

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


class FieldMetric(Metric):
    """Applies to each ScalarNode (leaf).

    A field metric *is* the comparison. Comparison metrics implement
    ``score(actual, expected)`` — a pure value-level primitive reused by array
    alignment (Stage 7). Metrics that need node context (e.g. presence) override
    ``compute(node)`` directly. ``compute`` delegates to ``score`` by default.
    """

    def compute(self, node: ScalarNode) -> float | dict[str, float]:
        return self.score(node.actual, node.expected)

    def score(self, actual: Any, expected: Any) -> float | dict[str, float]:
        raise NotImplementedError


class ObjectMetric(Metric):
    """Applies to each ObjectNode (root and nested)."""

    @abstractmethod
    def compute(self, node: ObjectNode) -> float | dict[str, float]: ...


class ArrayMetric(Metric):
    """Applies to each ArrayNode."""

    @abstractmethod
    def compute(self, node: ArrayNode) -> float | dict[str, float]: ...


class RootMetric(Metric):
    """Applies only to the root node (path == "$"); receives any EvalNode."""

    @abstractmethod
    def compute(self, node: EvalNode) -> float | dict[str, float]: ...


class NodeMetric(Metric):
    """Mixin for metrics spanning several node types.

    Override whichever of ``compute_scalar`` / ``compute_object`` /
    ``compute_array`` apply; the engine dispatches by node type.
    """


def get_metric_class(name: str) -> type:
    """Resolve a metric class by its ``name`` (e.g. ``"object_f1"``)."""
    if name not in _METRIC_REGISTRY:
        raise KeyError(
            f"Unknown metric: {name!r}. Known: {sorted(_METRIC_REGISTRY)}"
        )
    return _METRIC_REGISTRY[name]
