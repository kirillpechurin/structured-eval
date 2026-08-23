"""Verdicts for array metrics: aligned items → `(score, threshold, weight)`.

Concepts:

- TP (True Positive) — a `matched` entry, present on both sides;
- FP (False Positive) — a `spurious` entry, produced but not expected;
- FN (False Negative) — a `missing` entry, expected but not produced.

The verdicts feed `calculate.prf_counts`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from structured_eval.models.nodes.array_node import ArrayNode


def verdicts(node: ArrayNode, threshold: float) -> list[tuple[float, float, float]]:
    """`(representative, threshold, weight=1.0)` for each aligned item.

    Reads `node.matched`, not `node.items`: an element with no expected
    counterpart is already counted as spurious by `missing_spurious`, so
    grading it here would count it twice.

    Args:
        node: The array node whose aligned items to grade.
        threshold: The bar each item must clear.

    Returns:
        One verdict per aligned item.
    """
    return [(item.representative, threshold, 1.0) for item in node.matched]


def missing_spurious(node: ArrayNode) -> tuple[int, int]:
    """`(n_missing, n_spurious)` — the elements present on only one side."""
    return len(node.missing), len(node.spurious)
