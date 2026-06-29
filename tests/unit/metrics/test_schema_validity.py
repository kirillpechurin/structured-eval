"""SchemaValidity — validate the document against a pydantic model or JSON Schema.

Returns ``(score, extra)`` where ``extra["schema_errors"]`` buckets failures into
type_errors / missing_required / extra_fields.
"""

import pytest
from pydantic import BaseModel

from structured_eval import SchemaValidity

pytestmark = pytest.mark.unit


class Invoice(BaseModel):
    id: str
    total: float
    status: str


def test_valid_pydantic_document(tree_factory) -> None:
    root = tree_factory({"id": "1", "total": 100.0, "status": "paid"}, None)
    score, extra = SchemaValidity(Invoice).compute(root)
    assert score == 1.0
    assert extra["schema_errors"] == {
        "type_errors": [],
        "missing_required": [],
        "extra_fields": [],
    }


def test_wrong_type_flagged(tree_factory) -> None:
    root = tree_factory({"id": "1", "total": "not-a-float", "status": "paid"}, None)
    score, extra = SchemaValidity(Invoice).compute(root)
    assert score == 0.0
    assert "total" in extra["schema_errors"]["type_errors"]


def test_missing_required_flagged(tree_factory) -> None:
    root = tree_factory({"id": "1"}, None)
    score, extra = SchemaValidity(Invoice).compute(root)
    assert score == 0.0
    assert "total" in extra["schema_errors"]["missing_required"]


def test_accepts_raw_json_schema(tree_factory) -> None:
    schema = {
        "type": "object",
        "properties": {"id": {"type": "string"}, "n": {"type": "number"}},
        "required": ["id", "n"],
    }
    metric = SchemaValidity(schema)
    assert metric.compute(tree_factory({"id": "x", "n": 1}, None))[0] == 1.0
    assert metric.compute(tree_factory({"id": "x", "n": "bad"}, None))[0] == 0.0


def test_unsupported_schema_type_raises() -> None:
    from structured_eval.metrics.schema_validity.validator import SchemaValidator

    with pytest.raises(TypeError):
        SchemaValidator("not-a-schema").validate({})  # type: ignore[arg-type]
