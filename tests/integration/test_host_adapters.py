"""Host-framework adapters (deepeval / langsmith).

These import optional host libraries; the tests are skipped when the extras are
not installed. The framework-agnostic mapping is covered separately in
tests/unit/test_adapter.py (no host library needed).
"""

import pytest

from structured_eval import EvalConfig, ObjectF1

pytestmark = pytest.mark.integration


def test_deepeval_metric_scores_report():
    pytest.importorskip("deepeval")
    from deepeval.test_case import LLMTestCase

    from structured_eval.integrations.deepeval import StructuredMetric

    metric = StructuredMetric(config=EvalConfig(key_metric=ObjectF1()), threshold=0.8)
    tc = LLMTestCase(input="x", actual_output='{"a": 1}', expected_output='{"a": 1}')
    metric.measure(tc)
    assert metric.score == 1.0
    assert metric.is_successful()


def test_langsmith_evaluator_constructs():
    pytest.importorskip("langsmith")
    from structured_eval.integrations.langsmith import StructuredEvaluator

    evaluator = StructuredEvaluator(config=EvalConfig(key_metric=ObjectF1()))
    assert evaluator is not None
