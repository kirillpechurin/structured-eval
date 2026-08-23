"""The `mean_score` metric — the default representative score of any node."""

from __future__ import annotations

from typing import TYPE_CHECKING

from structured_eval.metrics.base import AnyNodeMetric

if TYPE_CHECKING:
    from structured_eval.models.nodes.base import EvalNode


class MeanScore(AnyNodeMetric):
    """A node's representative score: the arithmetic mean of its own metrics.

    The default `key_metric` of every node — the number that bubbles up to a
    parent's aggregation and, at the root, to `report.score`. Computed last, it
    averages the node's other metrics, itself excluded.

    - It never recurses into children. Cross-child aggregation is the job of the
      node's own metrics, one of which the engine always provides.
    - A node whose only metric opted out scores 0.0, so every node has a
      representative.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import MeanScore, TokenF1
        >>> from structured_eval.models import EvalConfig
        >>> report = evaluate({"a": "one two"}, {"a": "one three"},
        ...                   EvalConfig(metrics=[TokenF1(), MeanScore()]))
        >>> float(report.field_scores["a"].metrics["mean_score"])
        0.5
    """

    name = "mean_score"

    def compute(self, node: EvalNode) -> float:
        """Mean of the node's other metric values; 0.0 when it has none."""
        values = [
            float(v) for name, v in node.metric_results.items() if name != self.name
        ]
        return sum(values) / len(values) if values else 0.0
