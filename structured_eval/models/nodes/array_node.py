"""`ArrayNode` — a list position — and the alignment result it carries."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from structured_eval.models.config import ArrayStrategy  # noqa: TC001
from structured_eval.models.nodes.base import EvalNode


class ArrayMatchResult(BaseModel):
    """Alignment of an actual array against an expected array.

    A structural breakdown only. For precision / recall / F1 use the
    **value-aware** array metrics (`ArrayPrecision` / `ArrayRecall` / `ArrayF1`),
    which grade each matched element rather than just counting it.

    Attributes:
        strategy: The aligner that produced this result.
        matched: `(expected_idx, actual_idx)` pairs.
        missed: Expected indices with no actual counterpart (FN).
        spurious: Actual indices absent from expected (FP).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    strategy: ArrayStrategy
    matched: list[tuple[int, int]] = Field(default_factory=list)
    missed: list[int] = Field(default_factory=list)
    spurious: list[int] = Field(default_factory=list)


class ArrayNode(EvalNode):
    """A list node.

    The tree follows the document, not the alignment: with no expected list to
    align against there are simply no pairs. Two consequences:

    - `missing` indexes the **expected** list, `spurious` the actual one. A
      spurious element has a node; an expected element with no counterpart does
      not, because a node's path is an actual index.
    - Metrics read `matched` / `missing` / `spurious`, never `match_result`.
      That one is for the report.

    Attributes:
        match_result: The aligner's structural result, for the report.
        items: One node per **actual** element, in document order.
        matched: The subset of `items` paired with an expected element.
        missing: Expected indices with no actual counterpart (FN).
        spurious: Actual indices absent from expected (FP).
    """

    match_result: ArrayMatchResult | None = None
    items: list[EvalNode] = Field(default_factory=list)
    matched: list[EvalNode] = Field(default_factory=list)
    missing: list[int] = Field(default_factory=list)
    spurious: list[int] = Field(default_factory=list)
