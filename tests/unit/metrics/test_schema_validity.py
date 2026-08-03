"""SchemaValidity — validate the document against a pydantic model or JSON Schema.

Returns ``(score, extra)`` where ``extra["schema_errors"]`` buckets failures into
type_errors / missing_required / extra_fields, each naming the field it is about.
A model is validated as its JSON Schema, so both forms answer identically.
"""

import sys
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from structured_eval.metrics import SchemaValidity
from structured_eval.metrics.schema_validity.validator import (
    SchemaResult,
    SchemaValidator,
)
from structured_eval.models import EvalNode

pytestmark = pytest.mark.unit


class Invoice(BaseModel):
    id: str
    total: float
    status: str


class ClosedInvoice(BaseModel):
    """A schema that refuses unknown fields (the default silently drops them)."""

    model_config = ConfigDict(extra="forbid")

    id: str


def test_valid_pydantic_document(tree_factory: Callable[..., EvalNode]) -> None:
    root = tree_factory({"id": "1", "total": 100.0, "status": "paid"}, None)
    score, extra = SchemaValidity(Invoice).compute(root)
    assert score == 1.0
    assert extra["schema_errors"] == {
        "type_errors": [],
        "missing_required": [],
        "extra_fields": [],
    }


def test_wrong_type_flagged(tree_factory: Callable[..., EvalNode]) -> None:
    root = tree_factory({"id": "1", "total": "not-a-float", "status": "paid"}, None)
    score, extra = SchemaValidity(Invoice).compute(root)
    assert score == 0.0
    assert "total" in extra["schema_errors"]["type_errors"]


def test_missing_required_flagged(tree_factory: Callable[..., EvalNode]) -> None:
    root = tree_factory({"id": "1"}, None)
    score, extra = SchemaValidity(Invoice).compute(root)
    assert score == 0.0
    assert "total" in extra["schema_errors"]["missing_required"]


def test_an_unknown_field_is_reported_as_extra() -> None:
    # Only a schema that forbids extras can say so; the failure is bucketed
    # apart from a type error, because the field is not wrong — it is unasked for.
    result = SchemaValidator(ClosedInvoice).validate({"id": "1", "note": "hi"})
    assert not result.valid
    assert result.extra_fields == ["note"]
    assert result.type_errors == []


def test_accepts_raw_json_schema(tree_factory: Callable[..., EvalNode]) -> None:
    schema = {
        "type": "object",
        "properties": {"id": {"type": "string"}, "n": {"type": "number"}},
        "required": ["id", "n"],
    }
    metric = SchemaValidity(schema)
    assert metric.compute(tree_factory({"id": "x", "n": 1}, None))[0] == 1.0
    assert metric.compute(tree_factory({"id": "x", "n": "bad"}, None))[0] == 0.0


@pytest.mark.parametrize(
    ("document", "bucket", "value"),
    [
        ({"id": "x"}, "missing_required", ["n"]),
        ({"id": "x", "n": 1, "oops": 1, "wat": 2}, "extra_fields", ["oops", "wat"]),
        ({"id": "x", "n": 1, "x_ok": 1}, "extra_fields", []),
        ({"id": 1, "n": 1}, "type_errors", ["id"]),
    ],
    ids=["required", "additional-properties", "pattern-properties", "type"],
)
def test_a_failure_lands_in_the_right_bucket_naming_its_field(
    document: dict[str, Any], bucket: str, value: list[str]
) -> None:
    # `required` and `additionalProperties` fail the *object*, so jsonschema's
    # own location points at the container: the offending key is named only in
    # the message, and every rejected key at once. Reporting the container would
    # not tell the reader which field to go fix.
    schema = {
        "type": "object",
        "properties": {"id": {"type": "string"}, "n": {"type": "number"}},
        "patternProperties": {"^x_": {}},
        "required": ["id", "n"],
        "additionalProperties": False,
    }
    result = SchemaValidator(schema).validate(document)
    assert getattr(result, bucket) == value


@pytest.mark.parametrize(
    ("document", "bucket", "value"),
    [
        ({"lines": [{"sku": 1}]}, "type_errors", ["lines[0].sku"]),
        ({"lines": [{}]}, "missing_required", ["lines[0].sku"]),
        ({"lines": [{"sku": "A", "oops": 1}]}, "extra_fields", ["lines[0].oops"]),
    ],
    ids=["type", "required", "additional-properties"],
)
def test_a_failure_deep_in_the_document_is_addressed_by_its_path(
    document: dict[str, Any], bucket: str, value: list[str]
) -> None:
    # Spelled the way `flatten` spells it, so a schema error can be looked up in
    # `report.field_scores` — jsonschema's own sequence of keys and indices
    # would read `lines.0.sku`.
    schema = {
        "type": "object",
        "properties": {
            "lines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"sku": {"type": "string"}},
                    "required": ["sku"],
                    "additionalProperties": False,
                },
            }
        },
    }
    result = SchemaValidator(schema).validate(document)
    assert getattr(result, bucket) == value


@pytest.mark.parametrize(
    ("document", "bucket"),
    [
        ({"id": "1", "total": 100.0, "status": "paid"}, None),
        ({"id": "1", "total": 100.0}, "missing_required"),
        ({"id": "1", "total": "100", "status": "paid"}, "type_errors"),
    ],
    ids=["valid", "missing", "coerced-type"],
)
def test_a_model_and_its_json_schema_agree(document: Any, bucket: str | None) -> None:
    # One metric, one verdict: the model is validated as its JSON Schema, so how
    # the schema was expressed cannot change the answer. That makes it stricter
    # than `model_validate`, which would coerce `"100"` into a float and hide
    # the very error this metric is here to catch.
    from_model = SchemaValidator(Invoice).validate(document)
    from_schema = SchemaValidator(Invoice.model_json_schema()).validate(document)

    assert from_model.model_dump() == from_schema.model_dump()
    assert from_model.valid is (bucket is None)
    if bucket is not None:
        assert getattr(from_model, bucket)


@pytest.mark.parametrize(
    ("result", "rate"),
    [
        (SchemaResult(valid=True), None),
        (SchemaResult(valid=False, type_errors=["a"], total_fields=4), 0.25),
    ],
    ids=["no-fields", "one-of-four"],
)
def test_the_type_error_rate_needs_fields_to_divide_by(
    result: SchemaResult, rate: float | None
) -> None:
    # An empty schema has no denominator; 0/0 is not "no type errors".
    assert result.type_error_rate == rate


def test_unsupported_schema_type_raises() -> None:
    with pytest.raises(TypeError):
        SchemaValidator("not-a-schema").validate({})  # type: ignore[arg-type]


@pytest.mark.parametrize("schema", [Invoice, {"type": "object"}], ids=["model", "dict"])
def test_missing_extra_reports_the_install_hint(
    schema: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A model is validated as its JSON Schema, so the extra is needed for that
    # form too — and saying so beats an ImportError from an inner module.
    monkeypatch.setitem(sys.modules, "jsonschema", None)

    with pytest.raises(ImportError, match=r"structured-eval\[jsonschema\]"):
        SchemaValidator(schema).validate({"id": "1"})
