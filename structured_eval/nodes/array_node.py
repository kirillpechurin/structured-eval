from __future__ import annotations

from dataclasses import dataclass, field

from structured_eval.core.config import ArrayStrategy
from structured_eval.nodes.base import EvalNode


@dataclass
class ArrayMatchResult:
    """Alignment of an actual array against an expected array.

    ``matched`` are ``(expected_idx, actual_idx)`` pairs (TP); ``missed`` are
    expected indices with no actual counterpart (FN); ``spurious`` are actual
    indices absent from expected (FP). precision/recall/f1 derive from these.
    """

    strategy: ArrayStrategy
    matched: list[tuple[int, int]] = field(default_factory=list)
    missed: list[int] = field(default_factory=list)
    spurious: list[int] = field(default_factory=list)

    @property
    def precision(self) -> float:
        tp, fp = len(self.matched), len(self.spurious)
        return tp / (tp + fp) if (tp + fp) else 0.0

    @property
    def recall(self) -> float:
        tp, fn = len(self.matched), len(self.missed)
        return tp / (tp + fn) if (tp + fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class ArrayNode(EvalNode):
    """A list node. ``items`` are the per-element nodes after matching."""

    match_result: ArrayMatchResult | None = None
    items: list[EvalNode] = field(default_factory=list)
    metric_results: dict[str, float] = field(default_factory=dict)
