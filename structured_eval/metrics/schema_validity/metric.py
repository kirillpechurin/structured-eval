from __future__ import annotations

from typing import TYPE_CHECKING, Any

from structured_eval.metrics.base import RootMetric
from structured_eval.metrics.schema_validity.validator import SchemaValidator

if TYPE_CHECKING:
    from pydantic import BaseModel

    from structured_eval.models.nodes.base import EvalNode


class SchemaValidity(RootMetric):
    """Does the actual document validate against ``schema``? 1.0 / 0.0.

    ``schema`` is a Pydantic model class or a JSON Schema dict. Validation
    errors are returned as the result's ``extra["schema_errors"]`` — read via
    ``report.metrics["schema_validity"].extra_values("schema_errors")``.
    """

    name = "schema_validity"

    def __init__(self, schema: type[BaseModel] | dict[str, Any]):
        self.validator = SchemaValidator(schema)

    def compute(self, node: EvalNode) -> tuple[float, dict[str, Any]]:
        result = self.validator.validate(node.actual)
        return (1.0 if result.valid else 0.0), {
            "schema_errors": {
                "type_errors": result.type_errors,
                "missing_required": result.missing_required,
                "extra_fields": result.extra_fields,
            }
        }
