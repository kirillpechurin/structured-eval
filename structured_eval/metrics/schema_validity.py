from __future__ import annotations

from typing import Any

from structured_eval.metrics.protocol import RootMetric
from structured_eval.nodes.base import EvalNode


class SchemaValidity(RootMetric):
    """Does the actual document validate against ``schema``? 1.0 / 0.0.

    ``schema`` is a Pydantic model class or a JSON Schema dict. Validation
    errors are collected in ``self.schema_errors`` and surfaced by the engine
    into ``report.schema_errors``.
    """

    name = "schema_validity"

    def __init__(self, schema: Any):
        self.schema = schema
        self.schema_errors: list[str] = []

    def compute(self, node: EvalNode) -> float:
        from structured_eval.schema.validator import validate

        result = validate(node.actual, self.schema)
        self.schema_errors = (
            [f"type: {e}" for e in result.type_errors]
            + [f"missing: {m}" for m in result.missing_required]
            + [f"extra: {x}" for x in result.extra_fields]
        )
        return 1.0 if result.valid else 0.0
