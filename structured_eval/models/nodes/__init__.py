"""The node types the evaluation tree is built from.

`EvalNode` is the shared base; `ScalarNode`, `ObjectNode` and `ArrayNode` are
the three shapes a position in the document can take. Every node carries its own
metrics plus a `key_metric` standing in as its representative score.
"""

from structured_eval.models.nodes.array_node import ArrayMatchResult, ArrayNode
from structured_eval.models.nodes.base import EvalNode, NodeType
from structured_eval.models.nodes.object_node import ObjectNode
from structured_eval.models.nodes.scalar import ScalarNode

__all__ = [
    "ArrayMatchResult",
    "ArrayNode",
    "EvalNode",
    "NodeType",
    "ObjectNode",
    "ScalarNode",
]
