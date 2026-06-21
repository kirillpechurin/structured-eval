from __future__ import annotations

from structured_eval.metrics.base import Metric
from structured_eval.model.nodes.base import EvalNode


class MeanScore(Metric[EvalNode]):
    """A node's representative score: the arithmetic mean of its own metrics.

    The default ``key_metric`` of every node — the single number that bubbles up
    to a parent's aggregation and, at the root, to ``report.score``. It is
    computed **last**, so by the time it runs the node's other metrics already
    populate ``metric_results``; it averages those (excluding itself), without
    recursing into children — any cross-child aggregation is the job of the
    node's *own* metrics (``ObjectAccuracy`` / ``ObjectF1`` / ``ArrayAccuracy``),
    which the engine guarantees by defaulting one onto every node. Returns
    ``None`` only if a node somehow has no other metric (then it contributes
    nothing).
    """

    name = "mean_score"

    def compute(self, node: EvalNode) -> float | None:
        values = [float(v) for name, v in node.metric_results.items() if name != self.name]
        return sum(values) / len(values) if values else None
