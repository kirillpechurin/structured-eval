from __future__ import annotations

from typing import TYPE_CHECKING

from structured_eval.metrics.base import RootMetric

if TYPE_CHECKING:
    from structured_eval.models.nodes.base import EvalNode


class CoverageLeafScore(RootMetric):
    """Fraction of expected leaf fields that are present (non-null) in actual.

    Completeness across the whole document, independent of value correctness.
    Counts only leaves expected to have a value; a document expecting nothing
    is vacuously 1.0. (Array elements missed during alignment have no leaf
    node and are covered by the array metrics instead.)
    """

    name = "coverage_leaf_score"

    def compute(self, node: EvalNode) -> float:
        expected = covered = 0
        for leaf in node.leaves():
            if leaf.expected is not None:
                expected += 1
                if leaf.actual is not None:
                    covered += 1
        return covered / expected if expected else 1.0
