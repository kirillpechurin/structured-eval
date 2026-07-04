from __future__ import annotations

from typing import TYPE_CHECKING

from structured_eval.metrics.base import AnyNodeMetric

if TYPE_CHECKING:
    from structured_eval.models.nodes.base import EvalNode


class CompositeScore(AnyNodeMetric):
    """Weighted blend of other metrics already computed on the same node.

    Given ``weights={metric_name: weight}``, the score is the weighted mean of
    those metrics' values on the node::

        score = Σ wᵢ · metric_resultsᵢ        (weights normalized to sum 1.0)

    The referenced metrics must already be present in ``node.metric_results``,
    so list them in the node's ``metrics`` (or as cascaded ``config.metrics``)
    alongside ``CompositeScore``. As a representative it is best used as the
    node's ``key_metric``, which the engine runs **last** — by then every other
    metric on the node is computed.

    Only the metrics named in ``weights`` contribute; any other metric on the
    node is ignored, and a named metric that is absent contributes ``0``. The
    result is clamped to ``[0, 1]`` (each input is expected in ``[0, 1]``).
    """

    name = "composite_score"

    def __init__(self, weights: dict[str, float]) -> None:
        if not weights:
            raise ValueError("CompositeScore requires at least one metric weight")
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("Sum of weights must be > 0")
        self.weights: dict[str, float] = {m: w / total for m, w in weights.items()}

    def compute(self, node: EvalNode) -> float:
        total = sum(
            weight * float(node.metric_results[name])
            for name, weight in self.weights.items()
            if name in node.metric_results
        )
        return min(1.0, max(0.0, total))
