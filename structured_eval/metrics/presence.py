from __future__ import annotations

from typing import TYPE_CHECKING

from structured_eval.metrics.base import FieldMetric

if TYPE_CHECKING:
    from structured_eval.model.nodes.scalar import ScalarNode


class Presence(FieldMetric):
    """Was the field populated? 1.0 if present and non-null, else 0.0.

    A single-value check — it ignores ``expected`` and looks only at ``actual``
    (a missing key surfaces as ``None`` through the node). Overrides ``compute``
    rather than ``score`` since it is not a comparison of two values.
    """

    name = "presence"

    def compute(self, node: ScalarNode) -> float:
        return 1.0 if node.actual is not None else 0.0
