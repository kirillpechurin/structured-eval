"""How well one aligned element matches — used to gate array/object TPs.

A scalar's score is its match-criterion verdict; an object's is the soft mean
of its matched scalar fields; an array's is the soft mean of its element scores
(recursive). Self-contained over ``match_criterion`` / ``array_stats`` so it
shares the arithmetic with the accuracy metrics without importing them (which
would otherwise create an import cycle).
"""

from __future__ import annotations

from structured_eval.metrics._shared import match_criterion as mc
from structured_eval.model.nodes.array_node import ArrayNode
from structured_eval.model.nodes.base import EvalNode
from structured_eval.model.nodes.object_node import ObjectNode
from structured_eval.model.nodes.scalar import ScalarNode


def element_score(node: EvalNode) -> float:
    if isinstance(node, ScalarNode):
        return mc.field_verdict(node)[0]
    if isinstance(node, ObjectNode):
        verdicts = mc.matched_scalar_verdicts(node)
        denom = len(verdicts) + len(node.missing)
        if denom == 0:
            return 1.0
        return sum(score for score, _ in verdicts) / denom
    if isinstance(node, ArrayNode):
        n_missing = len(node.match_result.missed) if node.match_result else 0
        denom = len(node.items) + n_missing
        if denom == 0:
            return 1.0
        return sum(element_score(item) for item in node.items) / denom
    return 0.0
