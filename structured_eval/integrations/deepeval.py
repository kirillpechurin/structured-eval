"""deepeval adapter: structured-eval as a ``BaseMetric``.

Usage (requires ``structured-eval[deepeval]``)::

    from structured_eval.integrations.deepeval import StructuredMetric
    from deepeval import assert_test
    from deepeval.test_case import LLMTestCase

    metric = StructuredMetric(config=cfg, threshold=0.85)
    assert_test(LLMTestCase(input=..., actual_output=raw, expected_output=ref), [metric])

``actual_output``/``expected_output`` may be JSON strings or already-parsed
objects — ``evaluate`` handles both. ``report.score`` becomes ``metric.score``;
failing fields are summarised into ``metric.reason``. Importing this module
requires deepeval to be installed (the ``[deepeval]`` extra).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deepeval.metrics import BaseMetric

from structured_eval.api import evaluate
from structured_eval.integrations._adapter import verdict
from structured_eval.model.config import EvalConfig

if TYPE_CHECKING:
    from structured_eval.model.result import EvalReport


class StructuredMetric(BaseMetric):  # type: ignore[no-untyped-call]  # deepeval.__init_subclass__ has no annotations
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
        self.report: EvalReport | None = None

    def measure(self, test_case: Any, *args: Any, **kwargs: Any) -> float:
        self.report = evaluate(test_case.actual_output, test_case.expected_output, self.config)
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
