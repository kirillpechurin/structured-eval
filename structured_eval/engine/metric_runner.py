"""Phase 2 of the pipeline: computing every node's own metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from structured_eval.metrics.invoker import MetricInvoker
from structured_eval.models.metrics import MetricResult

if TYPE_CHECKING:
    from structured_eval.metrics.base import BaseMetric, MetricOutput
    from structured_eval.models.nodes.base import EvalNode


class MetricRunner:
    """Phase 2: compute each node's own metrics across the tree, in place.

    Every node carries the metrics resolved for it by `TreeBuilder`. They are
    computed **post-order** — children before their parent — so an aggregating
    parent reads its children's already-computed representative scores, and
    computation stays uniform and fully recursive at any nesting depth.

    Within a node the `key_metric` runs *last*: it is the representative score
    and its logic may depend on the node's other metrics (the default
    `MeanScore` averages them). A metric returning `None` (for instance
    `FieldFaithfulness` with no source to grade against) is skipped.

    Example:
        >>> from structured_eval.engine import MetricRunner, TreeBuilder
        >>> from structured_eval.models import EvalConfig, EvalContext
        >>> from structured_eval.utils import flatten
        >>> actual, expected = {"status": "paid"}, {"status": "due"}
        >>> context = EvalContext(
        ...     actual=actual,
        ...     expected=expected,
        ...     source=None,
        ...     flat_actual=flatten(actual),
        ...     flat_expected=flatten(expected),
        ...     config=EvalConfig()
        ... )
        >>> root, _ = TreeBuilder(context).build()
        >>> MetricRunner().run(root)
        >>> float(root.children["status"].metric_results["exact_match"])
        0.0
        >>> float(root.representative)          # the key metric, computed last
        0.0
    """

    def run(self, root: EvalNode) -> None:
        """Compute the metrics of every node under `root`, in place.

        Args:
            root: The root of the tree `TreeBuilder` produced.

        Example:
            >>> from structured_eval.engine import MetricRunner, TreeBuilder
            >>> from structured_eval.models import EvalConfig, EvalContext
            >>> from structured_eval.utils import flatten
            >>> actual, expected = {"status": "paid"}, {"status": "due"}
            >>> context = EvalContext(
            ...     actual=actual,
            ...     expected=expected,
            ...     source=None,
            ...     flat_actual=flatten(actual),
            ...     flat_expected=flatten(expected),
            ...     config=EvalConfig()
            ... )
            >>> root, _ = TreeBuilder(context).build()
            >>> MetricRunner().run(root) is None     # computed in place
            True
            >>> sorted(root.children["status"].metric_results)
            ['exact_match', 'mean_score']
        """
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
        """Coerce any `compute` return into `{key: MetricResult}`.

        A metric may attach structured `extra` regardless of how it shapes its
        score, so all five return shapes are accepted: `None` (skip), a bare
        value, a `dict` of sub-scores, a `MetricResult`, or a
        `(value | dict, extra)` tuple.

        Args:
            name: The metric's name, the key a single unnamed value lands under.
            result: Whatever the metric returned.

        Returns:
            The results keyed by report name — empty for `None`, one entry per
            sub-score for a `dict`, one entry otherwise. A tuple's `extra` is
            merged into every entry it produces.
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
