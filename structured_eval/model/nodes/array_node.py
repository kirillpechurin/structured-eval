from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from structured_eval.model.nodes.base import EvalNode

if TYPE_CHECKING:
    from structured_eval.model.config import ArrayStrategy


class ArrayMatchResult(BaseModel):
    """Alignment of an actual array against an expected array.

    A structural breakdown only: ``matched`` are ``(expected_idx, actual_idx)``
    pairs, ``missed`` are expected indices with no actual counterpart (FN),
    ``spurious`` are actual indices absent from expected (FP). For precision /
    recall / F1 use the **value-aware** array metrics (``ArrayPrecision`` /
    ``ArrayRecall`` / ``ArrayF1``), which grade each matched element rather than
    just counting it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    strategy: ArrayStrategy
    matched: list[tuple[int, int]] = Field(default_factory=list)
    missed: list[int] = Field(default_factory=list)
    spurious: list[int] = Field(default_factory=list)


class ArrayNode(EvalNode):
    """A list node. ``items`` are the per-element nodes after matching."""

    match_result: ArrayMatchResult | None = None
    items: list[EvalNode] = Field(default_factory=list)
