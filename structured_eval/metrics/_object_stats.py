"""Precision / recall / F1 arithmetic over resolved field verdicts.

Each matched scalar field is both a *predicted* and an *expected* field;
``spurious`` keys add to predicted (FP), ``missing`` keys add to expected (FN).
So a present-but-wrong field lowers both precision and recall. Nested
object/array children are graded at their own node and are not counted here.

``verdicts`` is a list of ``(score, threshold)`` from ``_match_criterion``.
In ``hard`` mode a field is a TP when ``score >= threshold`` (counts 1.0); in
``soft`` mode each field contributes its ``score`` fractionally (no threshold).
"""

from __future__ import annotations

Mode = str  # "hard" | "soft"


def prf_counts(
    verdicts: list[tuple[float, float]],
    n_missing: int,
    n_spurious: int,
    mode: Mode = "hard",
) -> tuple[float, float, float]:
    """Return ``(tp, predicted, expected)``; ``tp`` is float in soft mode."""
    matched = len(verdicts)
    predicted = matched + n_spurious
    expected = matched + n_missing
    if mode == "soft":
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
