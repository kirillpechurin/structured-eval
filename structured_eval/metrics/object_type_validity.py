from __future__ import annotations

from typing import TYPE_CHECKING

from structured_eval.metrics.base import ObjectMetric
from structured_eval.metrics.invoker import MetricInvoker
from structured_eval.metrics.type_match import TypeMatch

if TYPE_CHECKING:
    from structured_eval.model.nodes.object_node import ObjectNode


class ObjectTypeValidity(ObjectMetric):
    """Fraction of present fields that are type-valid.

    A structural sanity check independent of value correctness: of the fields
    present in both, how many carry the right JSON type. ``TypeMatch`` covers
    every JSON type, so this validates scalars (``"100"`` vs ``100``) *and*
    containers (a ``list`` where an object was expected) alike — a basic
    type check, not a deep one. An object with no present fields is vacuously
    1.0.
    """

    name = "object_type_validity"

    def __init__(self) -> None:
        self._type_match = MetricInvoker(TypeMatch())

    def compute(self, node: ObjectNode) -> float:
        present = node.matched
        if not present:
            return 1.0
        valid = sum(self._type_match.scalar_on_node(n) for n in present)
        return valid / len(present)
