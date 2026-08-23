"""The `field_faithfulness` metric — is a leaf value backed by the sample's source?"""

from __future__ import annotations

from typing import TYPE_CHECKING

from structured_eval.metrics.base import FieldMetric

if TYPE_CHECKING:
    from structured_eval.models.nodes.scalar import ScalarNode


class FieldFaithfulness(FieldMetric):
    """Is this leaf value grounded in the sample's `source`?

    An L1 check: a leaf scores 1.0 when its string form appears verbatim,
    case-insensitively, in the sample's `source`, else 0.0 for a hallucination.
    Cascade it and the engine does the rest — the hallucinated fields are the
    leaves scoring 0.0 in `report.metrics["field_faithfulness"].by_path`.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import FieldFaithfulness
        >>> from structured_eval.models import EvalConfig
        >>> report = evaluate({"vendor": "Acme", "city": "Berlin"}, None,
        ...                   EvalConfig(metrics=[FieldFaithfulness()]),
        ...                   source="Invoice from Acme, paid in full.")
        >>> float(report.metrics["field_faithfulness"].representative())
        0.5
    """

    name = "field_faithfulness"

    def compute(self, node: ScalarNode) -> float | None:
        """Check one leaf against the sample's grounding source.

        Args:
            node: The leaf being graded.

        Returns:
            1.0 when the value appears verbatim in `source`, 0.0 when it does
            not, and `None` for a leaf with no value to check.

        Raises:
            ValueError: If the sample carries no `source`. Faithfulness is
                undefined without one, so this is a configuration error rather
                than a metric the engine skips.
        """
        source = node.context.source
        if source is None:
            raise ValueError(
                "Faithfulness requires a grounding `source`; pass source=... to evaluate()"
            )
        actual = node.actual
        if actual is None:
            return None
        return 1.0 if str(actual).lower() in source.lower() else 0.0
