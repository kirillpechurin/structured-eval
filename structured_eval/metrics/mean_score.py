from __future__ import annotations

from structured_eval.metrics.base import Metric
from structured_eval.model.nodes.base import EvalNode


class MeanScore(Metric[EvalNode]):
    """A node's representative score: the arithmetic mean of its own metrics.

    The default ``key_metric`` of every node — the single number that bubbles
    up to a parent's aggregation and, at the root, to ``report.score``. It is
    computed last, so by the time it runs the node's other metrics already
    populate ``metric_results``; it averages those values. A node with no
    metrics of its own yields ``None`` (nothing to average) — the engine then
    falls back to the recursive ``structural_score`` via ``repr_score``.
    """

    name = "score"

    def compute(self, node: EvalNode) -> float | None:
        values = list(node.metric_results.values())
        if not values:
            return None
        return sum(values) / len(values)
