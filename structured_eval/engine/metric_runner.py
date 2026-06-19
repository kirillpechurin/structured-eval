from __future__ import annotations

from typing import Any

from structured_eval.metrics.base import BaseMetric, GenericMetric, Metric
from structured_eval.model.nodes.array_node import ArrayNode
from structured_eval.model.nodes.base import EvalNode
from structured_eval.model.nodes.object_node import ObjectNode
from structured_eval.model.nodes.scalar import ScalarNode

# Node class → the GenericMetric method that handles it.
_GENERIC_METHOD: dict[type, str] = {
    ScalarNode: "compute_scalar",
    ObjectNode: "compute_object",
    ArrayNode: "compute_array",
}


class MetricRunner:
    """Phase 2: compute each node's own metrics across the tree, in place.

    Every node carries the metrics resolved for it by ``TreeBuilder``; this
    phase walks the tree **post-order** (children before their parent), so an
    aggregating parent reads its children's already-computed representative
    scores — computation is uniform and fully recursive at any nesting depth.
    Within a node the ``key_metric`` runs *last*: it is the representative score
    and its logic may depend on the node's other metrics (the default
    ``MeanScore`` averages them). A metric returning ``None`` (e.g.
    ``Faithfulness`` without a source) is skipped.
    """

    def run(self, root: EvalNode) -> None:
        self._visit(root)

    def _visit(self, node: EvalNode) -> None:
        for child in node.children_nodes():
            self._visit(child)
        key_metric = node.key_metric
        for metric in node.metrics:
            if metric is key_metric:
                continue
            self._apply(metric, node)
        if key_metric is not None:
            self._apply(key_metric, node)

    def _apply(self, metric: BaseMetric, node: EvalNode) -> None:
        result = self._compute(metric, node)
        if result is None:
            return
        results: dict[str, Any] = node.metric_results
        if isinstance(result, dict):
            results.update(result)
        else:
            results[metric.name] = result

    @staticmethod
    def _compute(metric: BaseMetric, node: EvalNode) -> float | dict[str, float] | None:
        """Run ``metric`` on ``node`` — ``Metric.compute`` or the generic ``compute_<kind>``."""
        if isinstance(metric, GenericMetric):
            method = _GENERIC_METHOD.get(type(node))
            if method is None or not hasattr(metric, method):
                return None
            result: float | dict[str, float] | None = getattr(metric, method)(node)
            return result
        assert isinstance(metric, Metric)  # every non-generic metric has compute(node)
        return metric.compute(node)
