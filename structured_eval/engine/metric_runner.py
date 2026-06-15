from __future__ import annotations

from typing import Any

from structured_eval.metrics.base import (
    ArrayMetric,
    FieldMetric,
    Metric,
    ObjectMetric,
    RootMetric,
)
from structured_eval.model.nodes.array_node import ArrayNode
from structured_eval.model.nodes.base import EvalNode
from structured_eval.model.nodes.object_node import ObjectNode
from structured_eval.model.nodes.scalar import ScalarNode


class MetricRunner:
    """Phase 2: apply each requested metric across the tree, in place.

    Typed metric base classes dispatch by node type; ``RootMetric`` fires only
    on the root. ``NodeMetric`` (and custom classes) are duck-typed via
    ``compute_scalar`` / ``compute_object`` / ``compute_array``. A metric that
    returns ``None`` (e.g. ``Faithfulness`` without a source) is skipped.
    """

    _DUCK_METHOD = {
        ScalarNode: "compute_scalar",
        ObjectNode: "compute_object",
        ArrayNode: "compute_array",
        EvalNode: "compute_root"
    }

    def run(self, root: EvalNode, metrics: list[Metric]) -> None:
        for metric in metrics:
            for node in root.walk():
                result = self.apply(metric, node)
                if result is not None:
                    self._store(node, metric, result)

    def apply(self, metric: Metric, node: EvalNode) -> float | dict | None:
        """Run ``metric`` on ``node`` if it applies, else return ``None``."""
        if isinstance(metric, FieldMetric) and isinstance(node, ScalarNode):
            return metric.compute(node)
        if isinstance(metric, ObjectMetric) and isinstance(node, ObjectNode):
            return metric.compute(node)
        if isinstance(metric, ArrayMetric) and isinstance(node, ArrayNode):
            return metric.compute(node)
        if isinstance(metric, RootMetric) and node.path in ("$", ""):
            return metric.compute(node)

        method = self._DUCK_METHOD.get(type(node))
        if method and hasattr(metric, method):
            return getattr(metric, method)(node)
        return None

    @staticmethod
    def _store(node: EvalNode, metric: Metric, result: float | dict) -> None:
        results: dict[str, Any] = node.metric_results
        if isinstance(result, dict):
            results.update(result)
        else:
            results[metric.name] = result
