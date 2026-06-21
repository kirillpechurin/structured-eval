from __future__ import annotations

from typing import Any

from structured_eval.metrics.base import RootMetric
from structured_eval.metrics.faithfulness.substring import SubstringFaithfulness
from structured_eval.model.nodes.base import EvalNode


class Faithfulness(RootMetric):
    """Are the document's leaf values grounded in ``source``? (L1 substring.)

    Fraction of checkable leaf values that appear verbatim (case-insensitive)
    in the sample's ``source`` text. Fields marked ``FieldConfig(derived=True)``
    are excluded. The paths of unsupported values are returned as the result's
    ``extra["hallucinated_fields"]`` — read via
    ``report.metrics["faithfulness"].extra_values("hallucinated_fields")``.

    Requires a grounding ``source`` on the sample — faithfulness is undefined
    without one, so a missing ``source`` is a configuration error (``ValueError``)
    rather than a silently omitted metric.
    """

    name = "faithfulness"

    def __init__(self) -> None:
        self.processor = SubstringFaithfulness()

    def compute(self, node: EvalNode) -> tuple[float, dict[str, Any]]:
        source = node.context.source
        if source is None:
            raise ValueError(
                "Faithfulness requires a grounding `source`; pass source=... to evaluate()"
            )
        actual = node.actual
        score, hallucinated = self.processor.compute(
            actual if isinstance(actual, dict) else {}, source, node.context.config
        )
        return score, {"hallucinated_fields": hallucinated}
