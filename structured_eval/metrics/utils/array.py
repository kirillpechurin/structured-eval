"""Verdicts for array metrics: aligned items → ``(score, threshold, weight)``.

The verdicts feed ``calculate.prf_counts``; ``missed`` items are FN, ``spurious``
items FP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from structured_eval.models.nodes.array_node import ArrayNode


def verdicts(node: ArrayNode, threshold: float) -> list[tuple[float, float, float]]:
    """``(representative, threshold, weight=1.0)`` for each aligned item.

    ``node.matched``, not ``node.items``: an element with no expected
    counterpart is a false positive, already counted as ``spurious`` by
    ``missing_spurious`` — grading it here would count it twice.
    """
    return [(item.representative, threshold, 1.0) for item in node.matched]


def missing_spurious(node: ArrayNode) -> tuple[int, int]:
    """``(n_missing, n_spurious)`` — the elements present on only one side."""
    return len(node.missing), len(node.spurious)
