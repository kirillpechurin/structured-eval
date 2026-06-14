"""deepeval adapter: structured-eval as a ``BaseMetric``.

Usage (requires ``structured-eval[deepeval]``)::

    from structured_eval.integrations.deepeval import StructuredMetric
    from deepeval import assert_test
    from deepeval.test_case import LLMTestCase

    metric = StructuredMetric(config=cfg, threshold=0.85)
    assert_test(LLMTestCase(input=..., actual_output=raw, expected_output=ref), [metric])

``actual_output``/``expected_output`` may be JSON strings or already-parsed
objects — ``evaluate`` handles both. ``report.score`` becomes ``metric.score``;
failing fields are summarised into ``metric.reason``.
"""

from __future__ import annotations

from typing import Any

from structured_eval._evaluate import evaluate
from structured_eval.core.config import EvalConfig
from structured_eval.integrations._adapter import verdict

_metric_class: type | None = None


def _build_metric_class() -> type:
    from deepeval.metrics import BaseMetric  # lazy: only needed at use time

    class StructuredMetric(BaseMetric):
        """Field-level structured-output metric for deepeval."""

        def __init__(
            self,
            config: EvalConfig | None = None,
            threshold: float = 0.5,
            *,
            include_reason: bool = True,
        ) -> None:
            self.config = config or EvalConfig()
            self.threshold = threshold
            self.include_reason = include_reason
            self.score: float = 0.0
            self.success: bool = False
            self.reason: str | None = None
            self.report = None

        def measure(self, test_case: Any, *args: Any, **kwargs: Any) -> float:
            self.report = evaluate(
                test_case.actual_output, test_case.expected_output, self.config
            )
            score, success, reason = verdict(self.report, self.threshold)
            self.score = 0.0 if score is None else score
            self.success = success
            self.reason = reason if self.include_reason else None
            return self.score

        async def a_measure(self, test_case: Any, *args: Any, **kwargs: Any) -> float:
            return self.measure(test_case, *args, **kwargs)

        def is_successful(self) -> bool:
            return self.success

        @property
        def __name__(self) -> str:  # shown in deepeval output
            return "Structured Eval"

    return StructuredMetric


def __getattr__(name: str) -> Any:
    """Build the BaseMetric subclass lazily so the import stays dependency-free."""
    if name == "StructuredMetric":
        global _metric_class
        if _metric_class is None:
            _metric_class = _build_metric_class()
        return _metric_class
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
