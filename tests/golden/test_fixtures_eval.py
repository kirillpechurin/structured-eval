"""Golden regression tests: run evaluate() on dataset fixtures and assert the
headline numbers. These pin real end-to-end behaviour across representative
shapes (invoice / NER / tool-call / deep-nested) so refactors can't silently
shift scores.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from structured_eval import (
    ArrayF1,
    ArrayFieldConfig,
    ArrayStrategy,
    EvalConfig,
    FieldConfig,
    ObjectF1,
    ObjectFieldConfig,
    OverallLeafScore,
    evaluate,
)

pytestmark = pytest.mark.golden

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# ── invoice extraction (data-driven from JSON) ──────────────────────────────


@pytest.mark.parametrize("case", _load("invoices.json")["cases"], ids=lambda c: c["name"])
def test_invoice_object_f1(case):
    r = evaluate(case["actual"], case["expected"], config=EvalConfig(metrics=[ObjectF1()]))
    assert r.metrics["object_f1"].representative() == pytest.approx(case["expect_object_f1"])


# ── NER: list of typed spans, aligned by (text,label) ───────────────────────


def test_ner_array_by_key():
    actual = {
        "entities": [
            {"text": "Acme", "label": "ORG"},
            {"text": "Berlin", "label": "LOC"},
            {"text": "ghost", "label": "MISC"},
        ]
    }
    expected = {
        "entities": [
            {"text": "Berlin", "label": "LOC"},
            {"text": "Acme", "label": "ORG"},
            {"text": "2024", "label": "DATE"},
        ]
    }
    cfg = EvalConfig(
        fields={
            "entities": ArrayFieldConfig(
                strategy=ArrayStrategy.BY_KEY,
                params={"key": "text"},
                item=ObjectFieldConfig(fields={"text": FieldConfig(), "label": FieldConfig()}),
            )
        },
        metrics=[ArrayF1()],
    )
    r = evaluate(actual, expected, config=cfg)
    am = r.array_matches["entities"]
    # Acme + Berlin align (2 matched), ghost spurious, 2024 missed
    assert len(am.matched) == 2
    assert len(am.spurious) == 1
    assert len(am.missed) == 1
    # value-aware P/R/F1 come from the array metrics, not the match result
    assert r.field_scores["entities"].metrics["array_f1"] == pytest.approx(2 / 3)


# ── tool call: function name + nested args object ───────────────────────────


def test_tool_call_nested():
    actual = {"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}
    expected = {"name": "get_weather", "arguments": {"city": "Paris", "unit": "fahrenheit"}}
    cfg = EvalConfig(metrics=[ObjectF1(), OverallLeafScore()])
    r = evaluate(actual, expected, config=cfg)
    # name correct, arguments.city correct, arguments.unit wrong → 2/3 leaves
    assert r.metrics["overall_leaf_score"].representative() == pytest.approx(2 / 3)
    assert r.field_scores["arguments.unit"].score == 0.0


# ── deeply nested document ──────────────────────────────────────────────────


def test_deep_nested():
    actual = {"a": {"b": {"c": {"d": 1, "e": 2}}}}
    expected = {"a": {"b": {"c": {"d": 1, "e": 9}}}}
    r = evaluate(actual, expected, config=EvalConfig(metrics=[OverallLeafScore()]))
    assert r.field_scores["a.b.c.d"].score == 1.0
    assert r.field_scores["a.b.c.e"].score == 0.0
    assert r.metrics["overall_leaf_score"].representative() == pytest.approx(0.5)
