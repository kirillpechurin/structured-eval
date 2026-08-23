"""The `object_type_validity` metric — how many present fields have the right type."""

from __future__ import annotations

from typing import TYPE_CHECKING

from structured_eval.metrics.base import ObjectMetric
from structured_eval.metrics.invoker import MetricInvoker
from structured_eval.metrics.type_match import TypeMatch

if TYPE_CHECKING:
    from structured_eval.models.nodes.object_node import ObjectNode


class ObjectTypeValidity(ObjectMetric):
    """Fraction of present fields that are type-valid.

    A structural sanity check independent of whether the values are right: of
    the fields present on both sides, how many carry the right JSON type.

    A basic check, not a deep one.

    An object with no field present on both sides scores 1.0: the metric only
    asks whether what *is* there has the right type, and none of it is
    mistyped. Whether those fields should have been there is completeness —
    `ObjectRecall` and `CoverageLeafScore` measure that.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import ObjectTypeValidity
        >>> from structured_eval.models import EvalConfig
        >>> report = evaluate({"a": 1, "b": "x"}, {"a": 1, "b": 2},
        ...                   EvalConfig(metrics=[ObjectTypeValidity()]))
        >>> float(report.metrics["object_type_validity"].representative())
        0.5
    """

    name = "object_type_validity"

    def __init__(self, name: str | None = None) -> None:
        """Build the metric.

        Args:
            name: Per-instance report key.
        """
        super().__init__(name=name)
        self._type_match = MetricInvoker(TypeMatch())

    def compute(self, node: ObjectNode) -> float:
        """Fraction of present fields carrying the right JSON type."""
        present = node.matched
        if not present:
            return 1.0
        valid = sum(self._type_match.scalar_on_node(n) for n in present)
        return valid / len(present)
