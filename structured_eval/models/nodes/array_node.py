"""`ArrayNode` — a list position — and the alignment result it carries."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from structured_eval.models.config import ArrayStrategy  # noqa: TC001
from structured_eval.models.nodes.base import EvalNode


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
    """A list node.

    ``items`` holds one node per **actual** element, in document order — the
    tree follows the document, exactly as ``ObjectNode.children`` holds every
    key from either side. Alignment does not decide which nodes exist: with no
    expected list to align against (faithfulness / schema-only mode) there are
    no pairs at all, and element-level metrics would have nothing to score.

    ``matched`` is the subset of ``items`` paired with an expected element —
    what the comparison metrics grade, mirroring ``ObjectNode.matched``.
    ``missing`` / ``spurious`` are the indices present on only one side (FN /
    FP), as ``ObjectNode`` holds the keys present on only one side.

    ``missing`` indexes the **expected** list, ``spurious`` the actual one, and
    the asymmetry is deliberate: a spurious element has a node (its actual
    index is in ``items``), while an expected element with no counterpart stays
    unmaterialized — a node's path is an *actual* index, so there is no honest
    path to give it.

    Metrics read these three, never ``match_result``: the alignment result is
    what the report shows a user, not the tree's own vocabulary for who paired
    up with whom.
    """

    match_result: ArrayMatchResult | None = None
    items: list[EvalNode] = Field(default_factory=list)
    matched: list[EvalNode] = Field(default_factory=list)
    missing: list[int] = Field(default_factory=list)
    spurious: list[int] = Field(default_factory=list)
