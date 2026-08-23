"""The `coverage_leaf_score` metric — how much of the expected document is filled."""

from __future__ import annotations

from typing import TYPE_CHECKING

from structured_eval.metrics.base import RootMetric

if TYPE_CHECKING:
    from structured_eval.models.nodes.base import EvalNode


class CoverageLeafScore(RootMetric):
    """Fraction of expected leaf fields that are present (non-null) in actual.

    Completeness across the whole document, independent of whether the values
    are right. Counts only leaves expected to have a value, so a document
    expecting nothing is vacuously 1.0. Array elements missed during alignment
    have no leaf node and are the array metrics' business instead.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import CoverageLeafScore
        >>> from structured_eval.models import EvalConfig
        >>> report = evaluate({"a": 1, "b": None}, {"a": 1, "b": 2},
        ...                   EvalConfig(metrics=[CoverageLeafScore()]))
        >>> float(report.metrics["coverage_leaf_score"].representative())
        0.5
    """

    name = "coverage_leaf_score"

    def compute(self, node: EvalNode) -> float:
        """Share of expected leaves that carry a value in actual."""
        expected = covered = 0
        for leaf in node.leaves():
            if leaf.expected is not None:
                expected += 1
                if leaf.actual is not None:
                    covered += 1
        return covered / expected if expected else 1.0
