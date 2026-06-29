from __future__ import annotations

from typing import TYPE_CHECKING

from structured_eval.metrics.base import FieldMetric

if TYPE_CHECKING:
    from structured_eval.model.nodes.scalar import ScalarNode


class FieldFaithfulness(FieldMetric):
    """Is this leaf value grounded in the sample's ``source``? (L1 substring.)

    A per-field faithfulness check, true to the framework's "comparison is a
    metric" core: each scalar leaf scores ``1.0`` if its string form appears
    (case-insensitively) verbatim in ``source``, else ``0.0`` (a hallucination).
    Cascade it via ``EvalConfig(metrics=[FieldFaithfulness()])`` and the engine
    does the rest — aggregation is just the usual leaf roll-up
    (``MeanScore`` / ``OverallLeafScore``), and the hallucinated fields are the
    leaves scoring ``0.0`` (``report.metrics["field_faithfulness"].by_path``).

    Requires a grounding ``source`` on the sample — faithfulness is undefined
    without one, so a missing ``source`` is a configuration error
    (``ValueError``) rather than a silently omitted metric.
    """

    name = "field_faithfulness"

    def compute(self, node: ScalarNode) -> float | None:
        source = node.context.source
        if source is None:
            raise ValueError(
                "Faithfulness requires a grounding `source`; pass source=... to evaluate()"
            )
        actual = node.actual
        if actual is None:
            return None
        return 1.0 if str(actual).lower() in source.lower() else 0.0
