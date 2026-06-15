from __future__ import annotations

from pydantic import Field

from structured_eval.model.nodes.base import EvalNode


class ObjectNode(EvalNode):
    """A dict node.

    ``matched`` holds child nodes present in both actual and expected.
    ``missing`` / ``spurious`` hold keys present on only one side (FN / FP).
    ``children`` maps every child key to its node for tree traversal.
    """

    matched: list[EvalNode] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    spurious: list[str] = Field(default_factory=list)
    children: dict[str, EvalNode] = Field(default_factory=dict)
