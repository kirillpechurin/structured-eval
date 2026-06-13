from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from structured_eval.metrics.protocol import (
    ArrayMetric,
    FieldMetric,
    Metric,
    ObjectMetric,
    RootMetric,
)
from structured_eval.nodes.array_node import ArrayNode
from structured_eval.nodes.base import EvalNode
from structured_eval.nodes.object_node import ObjectNode
from structured_eval.nodes.scalar import ScalarNode

_DUCK_METHOD = {
    ScalarNode: "compute_scalar",
    ObjectNode: "compute_object",
    ArrayNode: "compute_array",
}


def apply_metric(metric: Metric, node: EvalNode) -> float | dict | None:
    """Run ``metric`` on ``node`` if it applies, else return ``None``.

    Typed metric base classes dispatch by node type; ``RootMetric`` only fires
    on the root. ``NodeMetric`` (and custom classes) are duck-typed via
    ``compute_scalar`` / ``compute_object`` / ``compute_array``.
    """
    if isinstance(metric, FieldMetric) and isinstance(node, ScalarNode):
        return metric.compute(node)
    if isinstance(metric, ObjectMetric) and isinstance(node, ObjectNode):
        return metric.compute(node)
    if isinstance(metric, ArrayMetric) and isinstance(node, ArrayNode):
        return metric.compute(node)
    if isinstance(metric, RootMetric) and node.path in ("$", ""):
        return metric.compute(node)

    method = _DUCK_METHOD.get(type(node))
    if method and hasattr(metric, method):
        return getattr(metric, method)(node)
    return None


def walk(node: EvalNode) -> Iterator[EvalNode]:
    """Depth-first traversal yielding every node in the tree."""
    yield node
    if isinstance(node, ObjectNode):
        for child in node.children.values():
            yield from walk(child)
    elif isinstance(node, ArrayNode):
        for item in node.items:
            yield from walk(item)


def _store(node: EvalNode, metric: Metric, result: float | dict) -> None:
    results: dict[str, Any] = node.metric_results
    if isinstance(result, dict):
        results.update(result)
    else:
        results[metric.name] = result


def run(root: EvalNode, metrics: list[Metric]) -> None:
    """Phase 2: apply each requested metric across the tree (in place)."""
    for metric in metrics:
        for node in walk(root):
            result = apply_metric(metric, node)
            if result is not None:
                _store(node, metric, result)
