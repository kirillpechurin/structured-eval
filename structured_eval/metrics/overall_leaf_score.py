"""The `overall_leaf_score` metric — weighted mean of leaf scores document-wide."""

from __future__ import annotations

from typing import TYPE_CHECKING

from structured_eval.metrics.base import RootMetric
from structured_eval.models.nodes.scalar import ScalarNode

if TYPE_CHECKING:
    from structured_eval.models.nodes.base import EvalNode


class OverallLeafScore(RootMetric):
    """Weighted mean of leaf match-criterion scores over the whole document.

    Each scalar field contributes its match-criterion verdict weighted by its
    configured `weight`. Missing expected leaves score 0, and a document with
    no leaves is vacuously 1.0.

    Leaf-style: it flattens to scalar leaves across the whole tree, where the
    object metrics aggregate a node's direct children.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import OverallLeafScore
        >>> from structured_eval.models import EvalConfig
        >>> report = evaluate({"a": 1, "b": 2}, {"a": 1, "b": 9},
        ...                   EvalConfig(metrics=[OverallLeafScore()]))
        >>> float(report.metrics["overall_leaf_score"].representative())
        0.5
    """

    name = "overall_leaf_score"

    def compute(self, node: EvalNode) -> float:
        """Weighted mean of every leaf's representative score."""
        total_weight = 0.0
        weighted = 0.0
        for leaf in node.leaves():
            assert isinstance(leaf, ScalarNode)
            total_weight += leaf.weight
            weighted += leaf.weight * leaf.representative
        return weighted / total_weight if total_weight else 1.0
