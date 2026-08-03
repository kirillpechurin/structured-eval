"""Verdicts for array metrics: aligned items → ``(score, threshold, weight)``.

An aligned item is graded by its representative score against a single
``threshold`` (hard) or contributes that score fractionally (soft) — mirroring
how object fields are graded. ``missed`` items are FN, ``spurious`` items FP.
The verdicts feed ``calculate.prf_counts``.

Array elements share one ``item`` config, so they carry no individual weights:
every item (and every missed/spurious slot) weighs ``1.0`` and array metrics are
effectively count-based.
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
