"""Precision / recall / F1 arithmetic over resolved field/item verdicts.

Each matched scalar field (or array item) is both a *predicted* and an
*expected* entry; ``spurious`` entries add to predicted (FP), ``missing`` ones
add to expected (FN). So a present-but-wrong entry lowers both precision and
recall. Nested object/array children are graded at their own node and are not
counted here.

A ``verdicts`` argument is a list of ``(score, threshold, weight)`` from
``structured_eval.metrics.utils.verdicts``. Each entry contributes its
``weight`` (``1.0`` by default → plain counts) rather than a flat 1: in
``GradingMode.HARD`` an entry is a TP when ``score >= threshold`` (counts its
weight); in ``GradingMode.SOFT`` it contributes ``weight * score`` (threshold
ignored). ``missing_weight`` / ``spurious_weight`` are the summed weights of the
FN / FP entries (counts when uniform). How those weights are derived is the
caller's choice (see ``WeightMode``).
"""

from __future__ import annotations

from enum import StrEnum


class GradingMode(StrEnum):
    """How a verdict counts toward true positives."""

    HARD = "hard"  # threshold gate: TP iff score >= threshold (counts its weight)
    SOFT = "soft"  # graded: weight * score contributes, no threshold


class WeightMode(StrEnum):
    """How a node's children contribute to its weighted aggregate.

    Extensible: more strategies (e.g. only first-level weights, or uniform per
    level) can be added without touching the arithmetic below.
    """

    NONE = "none"  # ignore configured weights — every child counts 1.0
    PROPORTIONAL = "proportional"  # weight each child by its configured ``weight``


def prf_counts(
    verdicts: list[tuple[float, float, float]],
    missing_weight: float,
    spurious_weight: float,
    mode: GradingMode = GradingMode.HARD,
) -> tuple[float, float, float]:
    """Return weighted ``(tp, predicted, expected)``; uniform weights → counts."""
    matched_weight = sum(weight for _, _, weight in verdicts)
    predicted = matched_weight + spurious_weight
    expected = matched_weight + missing_weight
    if mode == GradingMode.SOFT:
        tp = sum(weight * score for score, _, weight in verdicts)
    else:
        tp = sum(weight for score, threshold, weight in verdicts if score >= threshold)
    return tp, predicted, expected


def precision(tp: float, predicted: float, expected: float) -> float:
    if predicted == 0:
        return 1.0 if expected == 0 else 0.0  # empty object is vacuously precise
    return tp / predicted


def recall(tp: float, predicted: float, expected: float) -> float:
    if expected == 0:
        return 1.0 if predicted == 0 else 0.0
    return tp / expected


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) else 0.0
