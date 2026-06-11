from __future__ import annotations

from dataclasses import dataclass, field

from structured_eval.nodes.base import EvalNode


@dataclass
class ObjectNode(EvalNode):
    """A dict node.

    ``matched`` holds child nodes present in both actual and expected.
    ``missing`` / ``spurious`` hold keys present on only one side (FN / FP).
    ``children`` maps every child key to its node for tree traversal.
    """

    matched: list[EvalNode] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    spurious: list[str] = field(default_factory=list)
    children: dict[str, EvalNode] = field(default_factory=dict)
    metric_results: dict[str, float] = field(default_factory=dict)
