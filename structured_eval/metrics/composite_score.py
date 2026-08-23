"""The `composite_score` metric — a weighted blend of a node's other metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from structured_eval.metrics.base import AnyNodeMetric

if TYPE_CHECKING:
    from structured_eval.models.nodes.base import EvalNode


class CompositeScore(AnyNodeMetric):
    """Weighted blend of other metrics already computed on the same node.

    The weighted mean of the named metrics' values on the node, clamped to
    `[0, 1]`. Only the metrics named in `weights` contribute; a named metric
    that is absent contributes 0.

    The referenced metrics must already be computed, so list them alongside
    `CompositeScore` on the node. Best used as the node's `key_metric`, which
    the engine runs last.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import CompositeScore, ExactMatch, TokenF1
        >>> from structured_eval.models import EvalConfig, FieldConfig
        >>> blend = CompositeScore({"exact_match": 1, "token_f1": 1})
        >>> config = EvalConfig(
        ...     metrics=[ExactMatch(), TokenF1()],
        ...     fields={"a": FieldConfig(key_metric=blend)},
        ... )
        >>> report = evaluate({"a": "one two"}, {"a": "one three"}, config)
        >>> float(report.field_scores["a"].metrics["composite_score"])
        0.25
    """

    name = "composite_score"

    def __init__(self, weights: dict[str, float], name: str | None = None) -> None:
        """Set the blend.

        Args:
            weights: Metric name to weight; normalized to sum to 1.0.
            name: Per-instance report key.

        Raises:
            ValueError: If `weights` is empty or its values sum to 0 or less.
        """
        super().__init__(name=name)
        if not weights:
            raise ValueError("CompositeScore requires at least one metric weight")
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("Sum of weights must be > 0")
        self.weights: dict[str, float] = {m: w / total for m, w in weights.items()}

    def compute(self, node: EvalNode) -> float:
        """Weighted mean of the named metrics already on this node."""
        total = sum(
            weight * float(node.metric_results[name])
            for name, weight in self.weights.items()
            if name in node.metric_results
        )
        return min(1.0, max(0.0, total))
