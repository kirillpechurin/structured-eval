"""Tree traversal helper for root metrics (kept local to dodge import cycles)."""

from __future__ import annotations

from collections.abc import Iterator

from structured_eval.nodes.array_node import ArrayNode
from structured_eval.nodes.base import EvalNode
from structured_eval.nodes.object_node import ObjectNode
from structured_eval.nodes.scalar import ScalarNode


def leaves(node: EvalNode) -> Iterator[ScalarNode]:
    """Yield every scalar (leaf) node beneath ``node``."""
    if isinstance(node, ScalarNode):
        yield node
    elif isinstance(node, ObjectNode):
        for child in node.children.values():
            yield from leaves(child)
    elif isinstance(node, ArrayNode):
        for item in node.items:
            yield from leaves(item)
