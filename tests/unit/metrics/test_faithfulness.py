"""Unit tests for L1 substring faithfulness.

``SubstringFaithfulness.compute(actual, source, config) -> (score, hallucinated)``
checks each leaf value against the source text; the ``Faithfulness`` root metric
wraps it, reading ``source`` from the node context and returning ``None`` when
there is no source.
"""

from __future__ import annotations

import pytest

from structured_eval import EvalConfig, Faithfulness, FieldConfig
from structured_eval.metrics.faithfulness.substring import SubstringFaithfulness

pytestmark = pytest.mark.unit

SOURCE = "Invoice from Acme Corp, total amount 100.0 USD, status paid"


class TestSubstringFaithfulness:
    def test_all_found(self):
        score, hallucinated = SubstringFaithfulness().compute(
            {"vendor": "Acme Corp", "total": "100.0"}, SOURCE, EvalConfig()
        )
        assert score == 1.0
        assert hallucinated == []

    def test_hallucination_detected(self):
        score, hallucinated = SubstringFaithfulness().compute(
            {"vendor": "Globex"}, SOURCE, EvalConfig()
        )
        assert score == 0.0
        assert hallucinated == ["vendor"]

    def test_case_insensitive(self):
        score, _ = SubstringFaithfulness().compute({"vendor": "acme corp"}, SOURCE, EvalConfig())
        assert score == 1.0

    def test_partial(self):
        score, hallucinated = SubstringFaithfulness().compute(
            {"vendor": "Acme Corp", "city": "Berlin"}, SOURCE, EvalConfig()
        )
        assert score == pytest.approx(0.5)
        assert hallucinated == ["city"]

    def test_derived_field_skipped(self):
        cfg = EvalConfig(fields={"total": FieldConfig(derived=True)})
        # total is derived (not in source) but excluded → score from vendor only
        score, hallucinated = SubstringFaithfulness().compute(
            {"vendor": "Acme Corp", "total": "999"}, SOURCE, cfg
        )
        assert score == 1.0
        assert hallucinated == []

    def test_nested_and_list(self):
        score, hallucinated = SubstringFaithfulness().compute(
            {"meta": {"status": "paid"}, "tags": ["100.0", "ghost"]},
            SOURCE,
            EvalConfig(),
        )
        assert "tags[1]" in hallucinated
        assert 0.0 < score < 1.0

    def test_nothing_checkable_vacuous(self):
        score, hallucinated = SubstringFaithfulness().compute({}, SOURCE, EvalConfig())
        assert score == 1.0
        assert hallucinated == []


class TestFaithfulnessRootMetric:
    def test_requires_source(self, tree_factory):
        import pytest

        root = tree_factory({"vendor": "Globex"}, None)
        with pytest.raises(ValueError, match="source"):
            Faithfulness().compute(root)

    def test_score_and_hallucinations(self, tree_factory):
        root = tree_factory({"vendor": "Globex"}, None, source=SOURCE)
        metric = Faithfulness()
        score, extra = metric.compute(root)
        assert score == 0.0
        assert extra["hallucinated_fields"] == ["vendor"]
