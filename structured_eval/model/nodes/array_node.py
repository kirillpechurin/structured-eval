from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, computed_field

from structured_eval.model.config import ArrayStrategy
from structured_eval.model.nodes.base import EvalNode


class ArrayMatchResult(BaseModel):
    """Alignment of an actual array against an expected array.

    ``matched`` are ``(expected_idx, actual_idx)`` pairs (TP); ``missed`` are
    expected indices with no actual counterpart (FN); ``spurious`` are actual
    indices absent from expected (FP). precision/recall/f1 derive from these.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    strategy: ArrayStrategy
    matched: list[tuple[int, int]] = Field(default_factory=list)
    missed: list[int] = Field(default_factory=list)
    spurious: list[int] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def precision(self) -> float:
        tp, fp = len(self.matched), len(self.spurious)
        return tp / (tp + fp) if (tp + fp) else 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def recall(self) -> float:
        tp, fn = len(self.matched), len(self.missed)
        return tp / (tp + fn) if (tp + fn) else 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


class ArrayNode(EvalNode):
    """A list node. ``items`` are the per-element nodes after matching."""

    match_result: ArrayMatchResult | None = None
    items: list[EvalNode] = Field(default_factory=list)
