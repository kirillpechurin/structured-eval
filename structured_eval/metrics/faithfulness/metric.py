from __future__ import annotations

from structured_eval.metrics.base import RootMetric
from structured_eval.metrics.faithfulness.substring import SubstringFaithfulness
from structured_eval.model.nodes.base import EvalNode


class Faithfulness(RootMetric):
    """Are the document's leaf values grounded in ``source``? (L1 substring.)

    Fraction of checkable leaf values that appear verbatim (case-insensitive)
    in the sample's ``source`` text. Fields marked ``FieldConfig(derived=True)``
    are excluded. The paths of unsupported values are collected in
    ``self.hallucinated_fields`` and surfaced into ``report.hallucinated_fields``.

    Returns ``None`` (the engine then omits the metric) when the sample carries
    no ``source`` — faithfulness is undefined without a grounding text.
    """

    name = "faithfulness"

    def __init__(self) -> None:
        self.processor = SubstringFaithfulness()
        self.hallucinated_fields: list[str] = []

    def compute(self, node: EvalNode) -> float | None:  # type: ignore[override]
        source = node.context.source
        if source is None:
            return None
        actual = node.actual
        score, hallucinated = self.processor.compute(
            actual if isinstance(actual, dict) else {}, source, node.context.config
        )
        self.hallucinated_fields = hallucinated
        return score
