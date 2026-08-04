"""`ObjectNode` — a dict position in the evaluation tree."""

from __future__ import annotations

from pydantic import Field

from structured_eval.models.nodes.base import EvalNode


class ObjectNode(EvalNode):
    """A dict node.

    Attributes:
        matched: Child nodes present in both actual and expected.
        missing: Keys present only in expected (FN).
        spurious: Keys present only in actual (FP).
        children: Every child key mapped to its node, for tree traversal.
    """

    matched: list[EvalNode] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    spurious: list[str] = Field(default_factory=list)
    children: dict[str, EvalNode] = Field(default_factory=dict)
