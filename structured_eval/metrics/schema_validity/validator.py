"""Validation of a document against a Pydantic model or a JSON Schema dict."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, computed_field

if TYPE_CHECKING:
    from collections.abc import Iterable


class SchemaResult(BaseModel):
    """Outcome of validating actual against a schema.

    Attributes:
        valid: Whether the document satisfied the schema.
        type_errors: Fields whose value had the wrong type.
        missing_required: Required fields the document omitted.
        extra_fields: Fields the schema did not ask for.
        total_fields: How many properties the schema declares.
    """

    valid: bool
    type_errors: list[str] = []
    missing_required: list[str] = []
    extra_fields: list[str] = []
    total_fields: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def type_error_rate(self) -> float | None:
        """Share of declared fields with a type error; `None` if none declared."""
        if self.total_fields == 0:
            return None
        return len(self.type_errors) / self.total_fields


class SchemaValidator:
    """Validates a document against a Pydantic model class or JSON Schema dict.

    A model is converted to its JSON Schema first, so both forms are checked
    the same way. That makes validation stricter than `model_validate`, which
    coerces: `"100"` stays a string here, and reporting it as valid would hide
    exactly the error this metric exists to catch.
    """

    def __init__(self, schema: type[BaseModel] | dict[str, Any]):
        """Bind the schema documents are checked against.

        Args:
            schema: A pydantic model class or a JSON Schema dict.
        """
        self.schema = schema

    def validate(self, actual: Any) -> SchemaResult:
        """Check one document against the schema.

        Args:
            actual: The document to validate.

        Returns:
            The verdict, with the type errors, missing required fields and
            unasked-for fields it found.

        Raises:
            TypeError: If the bound schema is neither a pydantic model class
                nor a dict.
            ImportError: If the `jsonschema` extra is not installed.
        """
        schema = self._json_schema()
        total = len(schema.get("properties", {}))

        errors = list(self._validator(schema).iter_errors(actual))
        if not errors:
            return SchemaResult(valid=True, total_fields=total)

        type_errors: list[str] = []
        missing_required: list[str] = []
        extra_fields: list[str] = []

        for err in errors:
            # `required` and `additionalProperties` fail the *object*, so the
            # path locates the container and the offending key has to be named
            # separately; every other validator fails the value it points at.
            loc = self._path_of(err.absolute_path)
            if err.validator == "required":
                # err.message names the missing field directly
                field = err.message.split("'")[1] if "'" in err.message else ""
                missing_required.append(self._child_path(loc, field))
            elif err.validator == "additionalProperties":
                extra_fields += [
                    self._child_path(loc, key) for key in self._unexpected(err)
                ]
            else:
                # type and other validators (pattern, minLength, …) → type errors
                type_errors.append(loc or err.json_path)

        return SchemaResult(
            valid=False,
            type_errors=type_errors,
            missing_required=missing_required,
            extra_fields=extra_fields,
            total_fields=total,
        )

    # ── the schema, whichever way it was given ──────────────────────────────

    def _json_schema(self) -> dict[str, Any]:
        """The bound schema as a JSON Schema, converting a model if need be."""
        schema = self.schema
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema.model_json_schema()
        if isinstance(schema, dict):
            return schema
        raise TypeError(
            f"schema must be a Pydantic BaseModel subclass or a dict, got {type(schema)!r}"
        )

    @staticmethod
    def _validator(schema: dict[str, Any]) -> Any:
        """A jsonschema validator, imported at call time to keep the extra optional."""
        try:
            from jsonschema import Draft7Validator
        except ImportError as exc:
            raise ImportError(
                "jsonschema is required for schema validation. "
                "Install it with: pip install 'structured-eval[jsonschema]'"
            ) from exc
        return Draft7Validator(schema)

    # ── naming the field an error is about ──────────────────────────────────

    @staticmethod
    def _path_of(segments: Iterable[Any]) -> str:
        """A document path in this project's notation, e.g. `lines[0].sku`.

        Spelling it the way `flatten` does keeps `schema_errors` comparable
        with `report.field_scores`.

        Args:
            segments: The location jsonschema reported, as keys and indices.

        Returns:
            The same location as a document path.
        """
        path = ""
        for segment in segments:
            if isinstance(segment, int):
                path += f"[{segment}]"
            else:
                path = f"{path}.{segment}" if path else str(segment)
        return path

    @staticmethod
    def _child_path(parent: str, key: str) -> str:
        """The path of `key` inside the object at `parent` (maybe the root)."""
        return f"{parent}.{key}" if parent else key

    @staticmethod
    def _unexpected(error: Any) -> list[str]:
        """The keys `additionalProperties` rejected, in document order.

        jsonschema fails the object as a whole and names the offending keys
        only inside its message, so they are recomputed from the schema that
        rejected them — the report has to say *which* field was unasked for.

        Args:
            error: The jsonschema error that failed the object.

        Returns:
            The rejected keys, in document order.
        """
        schema = error.schema
        allowed = set(schema.get("properties", {}))
        patterns = [re.compile(p) for p in schema.get("patternProperties", {})]
        return [
            key
            for key in error.instance
            if key not in allowed and not any(p.search(key) for p in patterns)
        ]
