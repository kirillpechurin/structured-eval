from __future__ import annotations

from typing import TYPE_CHECKING, Any

from structured_eval.metrics.invoker import MetricInvoker
from structured_eval.models.metrics import MetricResult

if TYPE_CHECKING:
    from structured_eval.metrics.base import BaseMetric, MetricOutput
    from structured_eval.models.nodes.base import EvalNode


class MetricRunner:
    """Phase 2: compute each node's own metrics across the tree, in place.

    Every node carries the metrics resolved for it by ``TreeBuilder``. They are
    computed **post-order** — children before their parent — so an aggregating
    parent reads its children's already-computed representative scores, and
    computation stays uniform and fully recursive at any nesting depth.

    Within a node the ``key_metric`` runs *last*: it is the representative score
    and its logic may depend on the node's other metrics (the default
    ``MeanScore`` averages them). A metric returning ``None`` (e.g.
    ``FieldFaithfulness`` without a source) is skipped.

    Post-order plus "key_metric last" is not a style choice, it is the
    dependency graph written down: an aggregating metric reads
    ``child.representative``, which is the child's ``key_metric`` value, which
    in turn summarises the child's own metrics. The chain alternates between the
    two levels, so there is no cut that computes all the ordinary metrics first
    and all the representatives after.
    """

    def run(self, root: EvalNode) -> None:
        self._visit(root)

    def _visit(self, node: EvalNode) -> None:
        for child in node.children_nodes():
            self._visit(child)
        key_metric = node.key_metric
        for metric in node.metrics:
            if key_metric is not None and metric.name == key_metric.name:
                continue  # it is the representative; it runs last, just below
            self._apply(metric, node)
        if key_metric is not None:
            self._apply(key_metric, node)

    def _apply(self, metric: BaseMetric, node: EvalNode) -> None:
        result = MetricInvoker(metric).on_node(node)
        node.metric_results.update(self._normalize(metric.name, result))

    @staticmethod
    def _normalize(name: str, result: MetricOutput) -> dict[str, MetricResult]:
        """Coerce any ``compute`` return into ``{key: MetricResult}``.

        Accepts ``None`` (skip), a bare value, a ``dict`` of sub-scores, a
        ``MetricResult``, or a ``(value | dict, extra)`` tuple — so a metric can
        attach structured ``extra`` regardless of how it shapes its score. A
        tuple's ``extra`` is attached to every key it produces.
        """
        if result is None:
            return {}
        extra: dict[str, Any] = {}
        if isinstance(result, tuple):
            result, extra = result
        if isinstance(result, dict):
            return {
                k: MetricResult(v, {**getattr(v, "extra", {}), **extra})
                for k, v in result.items()
            }
        if isinstance(result, MetricResult):
            return {
                name: MetricResult(result, {**result.extra, **extra})
                if extra
                else result
            }
        return {name: MetricResult(result, extra)}
