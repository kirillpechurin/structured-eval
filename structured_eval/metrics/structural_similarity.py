"""The `structural_similarity` metric — Jaccard overlap of two documents' paths."""

from __future__ import annotations

from typing import TYPE_CHECKING

from structured_eval.metrics.base import RootMetric
from structured_eval.utils.flatten import extract_paths

if TYPE_CHECKING:
    from structured_eval.metrics.base import MetricOutput
    from structured_eval.models.nodes.base import EvalNode


class StructuralSimilarity(RootMetric):
    """Structural similarity of two documents — Jaccard over their paths.

    Jaccard overlap of the two documents' path sets, ignoring values — it
    answers whether the model produced the right skeleton. A path is enumerated
    for every dict key, list index and nested sub-path, containers and leaves
    alike; see `extract_paths`.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import StructuralSimilarity
        >>> from structured_eval.models import EvalConfig
        >>> report = evaluate({"a": 1}, {"a": 1, "b": 2},
        ...                   EvalConfig(metrics=[StructuralSimilarity()]))
        >>> float(report.metrics["structural_similarity"].representative())
        0.5
    """

    name = "structural_similarity"

    def compute(self, node: EvalNode) -> MetricOutput:
        """Jaccard overlap of the two documents' path sets."""
        paths_a = extract_paths(node.context.actual)
        paths_e = extract_paths(node.context.expected)

        if not paths_a and not paths_e:
            return 1.0
        if not paths_a or not paths_e:
            return 0.0

        return len(paths_a & paths_e) / len(paths_a | paths_e)
