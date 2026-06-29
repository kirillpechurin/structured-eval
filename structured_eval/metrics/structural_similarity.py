from __future__ import annotations

from typing import TYPE_CHECKING

from structured_eval.metrics.base import RootMetric
from structured_eval.utils.flatten import extract_paths

if TYPE_CHECKING:
    from structured_eval.metrics.base import MetricOutput
    from structured_eval.model.nodes.base import EvalNode


class StructuralSimilarity(RootMetric):
    """Structural similarity of two documents — Jaccard over their paths.

    Compares the *shape* of ``actual`` and ``expected``, ignoring values::

        |paths_actual ∩ paths_expected| / |paths_actual ∪ paths_expected|

    where a path is enumerated for every dict key, list index and nested
    sub-path (containers and leaves alike — see
    :func:`~structured_eval.utils.flatten.extract_paths`). Returns ``1.0``
    for identical structure (both empty → vacuously ``1.0``), ``0.0`` for no
    shared path, and a value in ``(0, 1)`` otherwise. A complement to the
    value-aware metrics: it answers "did the model produce the right skeleton"
    regardless of whether the values are correct.
    """

    name = "structural_similarity"

    def compute(self, node: EvalNode) -> MetricOutput:
        paths_a = extract_paths(node.context.actual)
        paths_e = extract_paths(node.context.expected)

        if not paths_a and not paths_e:
            return 1.0
        if not paths_a or not paths_e:
            return 0.0

        return len(paths_a & paths_e) / len(paths_a | paths_e)
