"""Unit tests for L1 substring faithfulness.

``FieldFaithfulness`` is a per-field metric: each scalar leaf scores 1.0 if its
string form is a substring of the sample's ``source`` (case-insensitive), else
0.0. Aggregation is the usual leaf roll-up; hallucinated fields are the leaves
scoring 0.0 (``report.metrics["field_faithfulness"].by_path``). A missing
``source`` is a configuration error (``ValueError``).
"""

import pytest

from structured_eval import EvalConfig, FieldFaithfulness, evaluate

pytestmark = pytest.mark.unit

SOURCE = "Invoice from Acme Corp, total amount 100.0 USD, status paid"
CFG = EvalConfig(metrics=[FieldFaithfulness()])


def _faith(actual, source=SOURCE):
    return evaluate(actual, None, CFG, source=source).metrics["field_faithfulness"]


class TestFieldFaithfulness:
    def test_all_found(self):
        mc = _faith({"vendor": "Acme Corp", "total": "100.0"})
        assert mc.mean() == 1.0
        assert _hallucinated(mc) == []

    def test_hallucination_detected(self):
        mc = _faith({"vendor": "Globex"})
        assert mc.mean() == 0.0
        assert _hallucinated(mc) == ["vendor"]

    def test_case_insensitive(self):
        assert _faith({"vendor": "acme corp"}).mean() == 1.0

    def test_partial(self):
        mc = _faith({"vendor": "Acme Corp", "city": "Berlin"})
        assert mc.mean() == pytest.approx(0.5)
        assert _hallucinated(mc) == ["city"]

    def test_nested_object(self):
        # nested object leaves materialize without expected and are checked
        mc = _faith({"meta": {"status": "paid"}, "vendor": "Acme Corp"})
        assert mc.mean() == 1.0
        assert "meta.status" in mc.by_path

    def test_array_items_need_expected(self):
        # Known limitation: in schema-only mode (expected=None) array items are
        # not aligned, so no item nodes exist and faithfulness can't reach them.
        # With an expected list the items materialize and are checked.
        # TODO: materialize actual elements without expected — roadmap.
        without = evaluate({"tags": ["100.0", "ghost"]}, None, CFG, source="value 100.0 here")
        assert "field_faithfulness" not in without.metrics  # no array item nodes
        with_exp = evaluate(
            {"tags": ["100.0", "ghost"]},
            {"tags": ["100.0", "x"]},
            CFG,
            source="value 100.0 here",
        ).metrics["field_faithfulness"]
        assert _hallucinated(with_exp) == ["tags[1]"]

    def test_null_value_skipped(self):
        # a null leaf has nothing to ground → not counted
        mc = _faith({"vendor": "Acme Corp", "city": None})
        assert mc.mean() == 1.0
        assert "city" not in mc.by_path

    def test_nothing_checkable_no_metric(self):
        # no leaves at all → the metric never runs and the key is absent
        r = evaluate({}, None, CFG, source=SOURCE)
        assert "field_faithfulness" not in r.metrics

    def test_requires_source(self):
        with pytest.raises(ValueError, match="source"):
            evaluate({"vendor": "Globex"}, None, CFG)


def _hallucinated(mc):
    return [p for p, v in mc.by_path.items() if float(v) == 0.0]
