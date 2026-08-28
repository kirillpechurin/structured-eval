"""Golden regression: `evaluate` over the dataset fixtures, headline numbers pinned.

The fixtures cover representative shapes — invoice, NER, tool call, deep-nested,
root array — so a refactor that shifts a score cannot pass silently.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from structured_eval import evaluate
from structured_eval.metrics import ArrayF1, ObjectF1, OverallLeafScore
from structured_eval.models import (
    ArrayFieldConfig,
    ArrayStrategy,
    EvalConfig,
    FieldConfig,
    ObjectFieldConfig,
)

pytestmark = pytest.mark.golden

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load(name: str) -> dict[str, Any]:
    """The fixture JSON document with that file name."""
    result: dict[str, Any] = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return result


# ── invoice extraction (data-driven from JSON) ──────────────────────────────


@pytest.mark.parametrize(
    "case", _load("invoices.json")["cases"], ids=lambda c: c["name"]
)
def test_invoice_object_f1(case: Any) -> None:
    """Invoice extraction: every case's pinned `object_f1`."""
    r = evaluate(
        case["actual"], case["expected"], config=EvalConfig(metrics=[ObjectF1()])
    )
    assert r.metrics["object_f1"].representative() == pytest.approx(
        case["expect_object_f1"]
    )


# ── NER: list of typed spans, aligned by (text,label) ───────────────────────


def test_ner_array_by_key() -> None:
    """NER spans aligned by text: 2 matched, 1 spurious, 1 missed."""
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
                item=ObjectFieldConfig(
                    fields={"text": FieldConfig(), "label": FieldConfig()}
                ),
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


def test_tool_call_nested() -> None:
    """A tool call with nested `arguments`: 2 of its 3 leaves are correct."""
    actual = {"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}
    expected = {
        "name": "get_weather",
        "arguments": {"city": "Paris", "unit": "fahrenheit"},
    }
    cfg = EvalConfig(metrics=[ObjectF1(), OverallLeafScore()])
    r = evaluate(actual, expected, config=cfg)
    # name correct, arguments.city correct, arguments.unit wrong → 2/3 leaves
    assert r.metrics["overall_leaf_score"].representative() == pytest.approx(2 / 3)
    assert r.field_scores["arguments.unit"].score == 0.0


# ── the document itself is an array ─────────────────────────────────────────


def test_root_array_of_records() -> None:
    # "Extract every line item" answers with a list, not an object wrapping one.
    # The root's path is a label, so its elements spell themselves without it —
    # `[0].sku`, the same way `flatten` writes them and `navigate` reads them.
    """A document that *is* an array: elements align by key, not by index."""
    actual = [
        {"sku": "B-7", "qty": 5},
        {"sku": "A-1", "qty": 99},
        {"sku": "Z-9", "qty": 1},
    ]
    expected = [
        {"sku": "A-1", "qty": 2},
        {"sku": "B-7", "qty": 5},
        {"sku": "C-3", "qty": 7},
    ]
    cfg = EvalConfig(
        root=ArrayFieldConfig(
            strategy=ArrayStrategy.BY_KEY,
            params={"key": "sku"},
            metrics=[ArrayF1()],
        )
    )
    r = evaluate(actual, expected, config=cfg)

    am = r.array_matches["$"]
    assert sorted(am.matched) == [(0, 1), (1, 0)]  # A-1 and B-7, despite the order
    assert am.missed == [2]  # C-3 never came back
    assert am.spurious == [2]  # Z-9 was invented

    # Elements resolve against their own counterpart, not whatever sits at the
    # same index: A-1 matched element 1, whose qty is wrong.
    assert r.field_scores["[1].sku"].score == 1.0
    assert r.field_scores["[1].qty"].score == 0.0
    assert r.field_scores["[0].qty"].score == 1.0

    # one of the two matched elements is fully correct → tp=1, predicted=3, expected=3
    assert r.metrics["array_f1"].representative() == pytest.approx(1 / 3)


# ── deeply nested document ──────────────────────────────────────────────────


def test_deep_nested() -> None:
    """A four-level document: leaf paths and scores survive the depth."""
    actual = {"a": {"b": {"c": {"d": 1, "e": 2}}}}
    expected = {"a": {"b": {"c": {"d": 1, "e": 9}}}}
    r = evaluate(actual, expected, config=EvalConfig(metrics=[OverallLeafScore()]))
    assert r.field_scores["a.b.c.d"].score == 1.0
    assert r.field_scores["a.b.c.e"].score == 0.0
    assert r.metrics["overall_leaf_score"].representative() == pytest.approx(0.5)
