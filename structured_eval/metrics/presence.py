"""The `presence` metric — was the field populated at all?"""

from __future__ import annotations

from typing import TYPE_CHECKING

from structured_eval.metrics.base import FieldMetric

if TYPE_CHECKING:
    from structured_eval.models.nodes.scalar import ScalarNode


class Presence(FieldMetric):
    """Was the field populated? 1.0 if present and non-null, else 0.0.

    A single-value check: it ignores `expected` and looks only at `actual`, so a
    missing key and an explicit `null` both score 0.0.

    Overrides `compute` rather than `score`, not being a comparison of two values.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import Presence
        >>> from structured_eval.models import EvalConfig
        >>> report = evaluate({"a": 1, "b": None}, {"a": 1, "b": 2},
        ...                   EvalConfig(metrics=[Presence()]))
        >>> float(report.metrics["presence"].representative())
        0.5
    """

    name = "presence"

    def compute(self, node: ScalarNode) -> float:
        """1.0 when the node carries a non-null value, else 0.0."""
        return 1.0 if node.actual is not None else 0.0
