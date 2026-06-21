from __future__ import annotations

from structured_eval.metrics.base import RootMetric
from structured_eval.model.nodes.base import EvalNode
from structured_eval.model.nodes.scalar import ScalarNode


class OverallScore(RootMetric):
    """Weighted mean of leaf match-criterion scores over the whole document.

    Each scalar field contributes its match-criterion verdict weighted by its
    configured ``weight`` (``FieldConfig.weight``). Missing expected leaves
    score 0; a document with no leaves is vacuously 1.0. The headline number.
    """

    name = "overall_score"

    def compute(self, node: EvalNode) -> float:
        total_weight = 0.0
        weighted = 0.0
        for leaf in node.leaves():  # TODO: Strange overall score
            assert isinstance(leaf, ScalarNode)
            total_weight += leaf.weight
            weighted += leaf.weight * leaf.representative
        return weighted / total_weight if total_weight else 1.0
