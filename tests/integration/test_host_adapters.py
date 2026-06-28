"""Host-framework adapters (deepeval / langsmith).

These import optional host libraries; the modules are skipped when the extras
are not installed. The framework-agnostic mapping is covered separately in
tests/unit/test_reporting.py::TestAdapter (no host library needed).
"""

import pytest

from structured_eval import EvalConfig, ObjectF1

pytestmark = pytest.mark.integration


class TestDeepeval:
    def test_metric_scores_report(self):
        pytest.importorskip("deepeval")
        from deepeval.test_case import LLMTestCase

        from structured_eval.integrations.deepeval import StructuredMetric

        metric = StructuredMetric(config=EvalConfig(key_metric=ObjectF1()), threshold=0.8)
        tc = LLMTestCase(
            input="x",
            actual_output='{"a": 1}',
            expected_output='{"a": 1}',
        )
        metric.measure(tc)
        assert metric.score == 1.0
        assert metric.is_successful()


class TestLangsmith:
    def test_evaluator_runs(self):
        pytest.importorskip("langsmith")
        from structured_eval.integrations.langsmith import StructuredEvaluator

        evaluator = StructuredEvaluator(config=EvalConfig(key_metric=ObjectF1()))
        assert evaluator is not None
