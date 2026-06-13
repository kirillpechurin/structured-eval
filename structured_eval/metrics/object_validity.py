from __future__ import annotations

from structured_eval.metrics.field.type_match import TypeMatch
from structured_eval.metrics.protocol import ObjectMetric
from structured_eval.nodes.object_node import ObjectNode
from structured_eval.nodes.scalar import ScalarNode


class ObjectValidity(ObjectMetric):
    """Fraction of present scalar fields that are type-valid.

    A structural sanity check independent of value correctness: of the fields
    present in both, how many carry the right JSON type (``"100"`` vs ``100``).
    An object with no present scalar fields is vacuously 1.0.
    """

    name = "object_validity"

    def __init__(self) -> None:
        self._type_match = TypeMatch()

    def compute(self, node: ObjectNode) -> float:
        present = [n for n in node.matched if isinstance(n, ScalarNode)]
        if not present:
            return 1.0
        valid = sum(self._type_match.score(n.actual, n.expected) for n in present)
        return valid / len(present)
