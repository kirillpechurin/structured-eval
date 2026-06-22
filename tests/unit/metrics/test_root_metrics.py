"""Unit tests for root (document-level) metrics: OverallLeafScore, CoverageLeafScore,
SchemaValidity. Run on the built tree root, mirroring how the engine fires them.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from structured_eval import (
    CoverageLeafScore,
    EvalConfig,
    FieldConfig,
    OverallLeafScore,
    SchemaValidity,
    TokenF1,
)

pytestmark = pytest.mark.unit


class Invoice(BaseModel):
    id: str
    total: float
    status: str


# ── OverallLeafScore ──────────────────────────────────────────────────────────


def test_overall_score_perfect(tree_factory):
    root = tree_factory({"a": 1, "b": 2}, {"a": 1, "b": 2})
    assert OverallLeafScore().compute(root) == 1.0


def test_overall_score_half(tree_factory):
    root = tree_factory({"a": 1, "b": 9}, {"a": 1, "b": 2})
    assert OverallLeafScore().compute(root) == pytest.approx(0.5)


def test_overall_score_weighted(tree_factory):
    cfg = EvalConfig(fields={"a": FieldConfig(weight=3.0), "b": FieldConfig(weight=1.0)})
    # a correct (w3), b wrong (w1) → 3/(3+1)
    root = tree_factory({"a": 1, "b": 9}, {"a": 1, "b": 2}, cfg)
    assert OverallLeafScore().compute(root) == pytest.approx(0.75)


def test_overall_score_uses_key_metric(tree_factory):
    cfg = EvalConfig(fields={"name": FieldConfig(metrics=[TokenF1()], key_metric=TokenF1())})
    root = tree_factory({"name": "the quick fox"}, {"name": "the quick brown fox"}, cfg)
    score = OverallLeafScore().compute(root)
    assert 0.0 < score < 1.0


def test_overall_score_empty_vacuous(tree_factory):
    root = tree_factory({}, {})
    assert OverallLeafScore().compute(root) == 1.0


# ── CoverageLeafScore ──────────────────────────────────────────────────────────────


def test_coverage_full(tree_factory):
    root = tree_factory({"a": 1, "b": 2}, {"a": 1, "b": 2})
    assert CoverageLeafScore().compute(root) == 1.0


def test_coverage_partial_missing(tree_factory):
    root = tree_factory({"a": 1}, {"a": 1, "b": 2})
    assert CoverageLeafScore().compute(root) == pytest.approx(0.5)


def test_coverage_null_not_covered(tree_factory):
    root = tree_factory({"a": 1, "b": None}, {"a": 1, "b": 2})
    assert CoverageLeafScore().compute(root) == pytest.approx(0.5)


def test_coverage_no_expected_vacuous(tree_factory):
    root = tree_factory({"a": 1}, {})
    assert CoverageLeafScore().compute(root) == 1.0


# ── SchemaValidity ────────────────────────────────────────────────────────


def test_schema_valid_pydantic(tree_factory):
    metric = SchemaValidity(Invoice)
    root = tree_factory({"id": "1", "total": 100.0, "status": "paid"}, None)
    score, extra = metric.compute(root)
    assert score == 1.0
    assert extra["schema_errors"] == []


def test_schema_invalid_type(tree_factory):
    metric = SchemaValidity(Invoice)
    root = tree_factory({"id": "1", "total": "not-a-float", "status": "paid"}, None)
    score, extra = metric.compute(root)
    assert score == 0.0
    assert any("total" in e for e in extra["schema_errors"])


def test_schema_invalid_missing(tree_factory):
    metric = SchemaValidity(Invoice)
    root = tree_factory({"id": "1"}, None)
    score, extra = metric.compute(root)
    assert score == 0.0
    assert any("missing" in e for e in extra["schema_errors"])


def test_schema_jsonschema_dict(tree_factory):
    schema = {
        "type": "object",
        "properties": {"id": {"type": "string"}, "n": {"type": "number"}},
        "required": ["id", "n"],
    }
    metric = SchemaValidity(schema)
    assert metric.compute(tree_factory({"id": "x", "n": 1}, None))[0] == 1.0
    assert metric.compute(tree_factory({"id": "x", "n": "bad"}, None))[0] == 0.0


def test_schema_bad_type_raises():
    from structured_eval.metrics.schema_validity.validator import SchemaValidator

    with pytest.raises(TypeError):
        SchemaValidator("not-a-schema").validate({})
