"""The `schema_validity` metric — does the document validate against a schema?"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from structured_eval.metrics.base import RootMetric
from structured_eval.metrics.schema_validity.validator import SchemaValidator

if TYPE_CHECKING:
    from pydantic import BaseModel

    from structured_eval.models.nodes.base import EvalNode


class SchemaValidity(RootMetric):
    """Does the actual document validate against `schema`? 1.0 or 0.0.

    Validation errors ride on the result's `extra["schema_errors"]`.
    Read them back via
    `report.metrics["schema_validity"].extra_values("schema_errors")`.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import SchemaValidity
        >>> from structured_eval.models import EvalConfig
        >>> schema = {"type": "object", "required": ["total"]}
        >>> config = EvalConfig(metrics=[SchemaValidity(schema)])
        >>> float(evaluate({"total": 10}, None, config)
        ...       .metrics["schema_validity"].representative())
        1.0
        >>> float(evaluate({"other": 10}, None, config)
        ...       .metrics["schema_validity"].representative())
        0.0
    """

    name = "schema_validity"

    def __init__(
        self, schema: type[BaseModel] | dict[str, Any], name: str | None = None
    ):
        """Bind the schema documents are checked against.

        Args:
            schema: A pydantic model class or a JSON Schema dict.
            name: Per-instance report key.
        """
        super().__init__(name=name)
        self.validator = SchemaValidator(schema)

    def compute(self, node: EvalNode) -> tuple[float, dict[str, Any]]:
        """1.0 when the document validates, else 0.0, plus the errors found."""
        result = self.validator.validate(node.actual)
        return (1.0 if result.valid else 0.0), {
            "schema_errors": {
                "type_errors": result.type_errors,
                "missing_required": result.missing_required,
                "extra_fields": result.extra_fields,
            }
        }
