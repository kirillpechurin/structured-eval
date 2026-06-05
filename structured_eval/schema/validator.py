from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError


@dataclass(frozen=True)
class SchemaResult:
    """Outcome of validating actual against a schema."""

    valid: bool
    type_errors: list[str] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    extra_fields: list[str] = field(default_factory=list)
    total_fields: int = 0

    @property
    def type_error_rate(self) -> float | None:
        if self.total_fields == 0:
            return None
        return len(self.type_errors) / self.total_fields


def validate(actual: Any, schema: Any) -> SchemaResult:
    """Validate actual against a Pydantic model class or a JSON Schema dict.

    Args:
        actual: The value to validate (typically a dict from LLM output).
        schema: Either a Pydantic BaseModel subclass or a JSON Schema dict.

    Raises:
        TypeError: If schema is neither a Pydantic model class nor a dict.
    """
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return _validate_pydantic(actual, schema)
    if isinstance(schema, dict):
        return _validate_jsonschema(actual, schema)
    raise TypeError(f"schema must be a Pydantic BaseModel subclass or a dict, got {type(schema)!r}")


# ── Pydantic ──────────────────────────────────────────────────────────────────


def _validate_pydantic(actual: Any, model: type[BaseModel]) -> SchemaResult:
    total = len(model.model_fields)
    try:
        model.model_validate(actual)
        return SchemaResult(valid=True, total_fields=total)
    except ValidationError as exc:
        type_errors: list[str] = []
        missing_required: list[str] = []
        extra_fields: list[str] = []

        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"])
            kind = err["type"]
            if kind == "missing":
                missing_required.append(loc)
            elif kind == "extra_forbidden":
                extra_fields.append(loc)
            else:
                type_errors.append(loc)

        return SchemaResult(
            valid=False,
            type_errors=type_errors,
            missing_required=missing_required,
            extra_fields=extra_fields,
            total_fields=total,
        )


# ── JSON Schema ───────────────────────────────────────────────────────────────


def _validate_jsonschema(actual: Any, schema: dict[str, Any]) -> SchemaResult:
    try:
        from jsonschema import Draft7Validator
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "jsonschema is required for dict-schema validation. "
            "Install it with: pip install 'structured-eval[jsonschema]'"
        ) from exc

    total = len(schema.get("properties", {}))
    validator = Draft7Validator(schema)
    errors = list(validator.iter_errors(actual))

    if not errors:
        return SchemaResult(valid=True, total_fields=total)

    type_errors: list[str] = []
    missing_required: list[str] = []
    extra_fields: list[str] = []

    for err in errors:
        loc = ".".join(str(p) for p in err.absolute_path) if err.absolute_path else err.json_path
        if err.validator == "required":
            # err.message names the missing field directly
            missing_field = err.message.split("'")[1] if "'" in err.message else loc
            missing_required.append(missing_field)
        elif err.validator == "additionalProperties":
            extra_fields.append(loc or err.message)
        elif err.validator == "type":
            type_errors.append(loc)
        # other validators (pattern, minLength, …) count as type errors
        else:
            type_errors.append(loc)

    return SchemaResult(
        valid=False,
        type_errors=type_errors,
        missing_required=missing_required,
        extra_fields=extra_fields,
        total_fields=total,
    )
