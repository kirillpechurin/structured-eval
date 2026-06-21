"""Precision / recall / F1 arithmetic over resolved field/item verdicts.

Each matched scalar field (or array item) is both a *predicted* and an
*expected* entry; ``spurious`` entries add to predicted (FP), ``missing`` ones
add to expected (FN). So a present-but-wrong entry lowers both precision and
recall. Nested object/array children are graded at their own node and are not
counted here.

A ``verdicts`` argument is a list of ``(score, threshold)`` from
``structured_eval.metrics.utils.verdicts``. In ``MatchMode.HARD`` an entry is a
TP when ``score >= threshold`` (counts 1.0); in ``MatchMode.SOFT`` each entry
contributes its ``score`` fractionally (the threshold is ignored).
"""

from __future__ import annotations

from enum import StrEnum


class MatchMode(StrEnum):
    """How a verdict counts toward true positives."""

    HARD = "hard"  # threshold gate: TP iff score >= threshold (counts 1.0)
    SOFT = "soft"  # graded: the score contributes fractionally, no threshold


def prf_counts(
    verdicts: list[tuple[float, float]],
    n_missing: int,
    n_spurious: int,
    mode: MatchMode = MatchMode.HARD,
) -> tuple[float, float, float]:
    """Return ``(tp, predicted, expected)``; ``tp`` is float in soft mode."""
    matched = len(verdicts)
    predicted = matched + n_spurious
    expected = matched + n_missing
    if mode == MatchMode.SOFT:
        tp = sum(score for score, _ in verdicts)
    else:
        tp = sum(1.0 for score, threshold in verdicts if score >= threshold)
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
