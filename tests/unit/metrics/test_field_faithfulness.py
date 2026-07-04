"""FieldFaithfulness — L1 substring grounding of each scalar leaf in the source.

Each leaf scores 1.0 if its string form is a substring of the sample's
``source`` (case-insensitive), else 0.0. Hallucinated fields are the leaves
scoring 0.0 (``report.metrics["field_faithfulness"].by_path``). A missing
``source`` is a configuration error (``ValueError``).
"""

from typing import Any

import pytest

from structured_eval import evaluate
from structured_eval.metrics import FieldFaithfulness
from structured_eval.models import EvalConfig, MetricCollection

pytestmark = pytest.mark.unit

SOURCE = "Invoice from Acme Corp, total amount 100.0 USD, status paid"
CFG = EvalConfig(metrics=[FieldFaithfulness()])


def _faith(actual: Any, source: str = SOURCE) -> MetricCollection:
    return evaluate(actual, None, CFG, source=source).metrics["field_faithfulness"]


def _hallucinated(mc: MetricCollection) -> list[str]:
    return [p for p, v in mc.by_path.items() if float(v) == 0.0]


def test_all_leaves_grounded() -> None:
    mc = _faith({"vendor": "Acme Corp", "total": "100.0"})
    assert mc.mean() == 1.0
    assert _hallucinated(mc) == []


def test_hallucination_detected() -> None:
    mc = _faith({"vendor": "Globex"})
    assert mc.mean() == 0.0
    assert _hallucinated(mc) == ["vendor"]


def test_case_insensitive() -> None:
    assert _faith({"vendor": "acme corp"}).mean() == 1.0


def test_partial_grounding() -> None:
    mc = _faith({"vendor": "Acme Corp", "city": "Berlin"})
    assert mc.mean() == pytest.approx(0.5)
    assert _hallucinated(mc) == ["city"]


def test_nested_object_leaves_checked() -> None:
    mc = _faith({"meta": {"status": "paid"}, "vendor": "Acme Corp"})
    assert mc.mean() == 1.0
    assert "meta.status" in mc.by_path


def test_array_items_need_expected() -> None:
    # Known limitation: in schema-only mode (expected=None) array items are not
    # aligned, so no item nodes exist and faithfulness can't reach them. With an
    # expected list the items materialize and are checked.
    # TODO: materialize actual elements without expected — roadmap.
    without = evaluate(
        {"tags": ["100.0", "ghost"]}, None, CFG, source="value 100.0 here"
    )
    assert "field_faithfulness" not in without.metrics  # no array item nodes
    with_exp = evaluate(
        {"tags": ["100.0", "ghost"]},
        {"tags": ["100.0", "x"]},
        CFG,
        source="value 100.0 here",
    ).metrics["field_faithfulness"]
    assert _hallucinated(with_exp) == ["tags[1]"]


def test_null_leaf_skipped() -> None:
    # a null leaf has nothing to ground → not counted
    mc = _faith({"vendor": "Acme Corp", "city": None})
    assert mc.mean() == 1.0
    assert "city" not in mc.by_path


def test_nothing_checkable_means_no_metric() -> None:
    # no leaves at all → the metric never runs and the key is absent
    r = evaluate({}, None, CFG, source=SOURCE)
    assert "field_faithfulness" not in r.metrics


def test_requires_source() -> None:
    with pytest.raises(ValueError, match="source"):
        evaluate({"vendor": "Globex"}, None, CFG)
