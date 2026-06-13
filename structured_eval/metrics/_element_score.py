"""How well one aligned element matches — used to gate array TPs.

A scalar's score is its match-criterion verdict; an object's is its accuracy;
an array's is its accuracy (recursive). Imports are local to avoid an import
cycle with the object/array accuracy metrics.
"""

from __future__ import annotations

from structured_eval.nodes.array_node import ArrayNode
from structured_eval.nodes.base import EvalNode
from structured_eval.nodes.object_node import ObjectNode
from structured_eval.nodes.scalar import ScalarNode


def element_score(node: EvalNode) -> float:
    if isinstance(node, ScalarNode):
        from structured_eval.metrics._match_criterion import field_verdict

        return field_verdict(node)[0]
    if isinstance(node, ObjectNode):
        from structured_eval.metrics.object_accuracy import ObjectAccuracy

        return ObjectAccuracy().compute(node)
    if isinstance(node, ArrayNode):
        from structured_eval.metrics.array_accuracy import ArrayAccuracy

        return ArrayAccuracy().compute(node)
    return 0.0
