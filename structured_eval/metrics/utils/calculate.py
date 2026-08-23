"""Precision / recall / F1 arithmetic over resolved field/item verdicts.

Concepts:

- TP (True Positive) — a `matched` entry, present on both sides;
- FP (False Positive) — a `spurious` entry, produced but not expected;
- FN (False Negative) — a `missing` entry, expected but not produced.

Each matched scalar field (or array item) is both a predicted and an expected
entry, so a present-but-wrong one lowers precision and recall alike. Nested
object and array children are graded at their own node, not counted here.
"""

from __future__ import annotations

from enum import StrEnum


class GradingMode(StrEnum):
    """How a verdict counts toward true positives.

    Attributes:
        HARD: Threshold gate — a TP iff the score clears the bar, counting its
            full weight.
        SOFT: Graded — `weight * score` contributes, with no threshold.
    """

    HARD = "hard"
    SOFT = "soft"


class WeightMode(StrEnum):
    """How a node's children contribute to its weighted aggregate.

    Attributes:
        NONE: Ignore the configured weights — every child counts 1.0.
        PROPORTIONAL: Weight each child by its configured `weight`.
    """

    NONE = "none"
    PROPORTIONAL = "proportional"


def prf_counts(
    verdicts: list[tuple[float, float, float]],
    missing_weight: float,
    spurious_weight: float,
    mode: GradingMode = GradingMode.HARD,
) -> tuple[float, float, float]:
    """Return weighted `(tp, predicted, expected)`; uniform weights → counts."""
    matched_weight = sum(weight for _, _, weight in verdicts)
    predicted = matched_weight + spurious_weight
    expected = matched_weight + missing_weight
    if mode == GradingMode.SOFT:
        tp = sum(weight * score for score, _, weight in verdicts)
    else:
        tp = sum(weight for score, threshold, weight in verdicts if score >= threshold)
    return tp, predicted, expected


def precision(tp: float, predicted: float, expected: float) -> float:
    """`tp / predicted`; an empty prediction is vacuously precise."""
    if predicted == 0:
        return 1.0 if expected == 0 else 0.0  # empty object is vacuously precise
    return tp / predicted


def recall(tp: float, predicted: float, expected: float) -> float:
    """`tp / expected`; expecting nothing is vacuously complete."""
    if expected == 0:
        return 1.0 if predicted == 0 else 0.0
    return tp / expected


def f1(p: float, r: float) -> float:
    """Harmonic mean of precision and recall; 0.0 when both are 0."""
    return 2 * p * r / (p + r) if (p + r) else 0.0
