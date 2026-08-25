"""deepeval adapter: structured-eval as a deepeval `BaseMetric`.

`StructuredMetric` scores a test case field by field instead of pass/fail:
`report.score` becomes `metric.score`, and the failing fields are summarised
into `metric.reason`. Requires `structured-eval[deepeval]`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    from deepeval.metrics import BaseMetric
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "deepeval is required for this integration. "
        "Install it with: pip install structured-eval[deepeval]"
    ) from exc

from structured_eval.api import evaluate
from structured_eval.integrations._adapter import verdict
from structured_eval.models.config import EvalConfig

if TYPE_CHECKING:
    from structured_eval.models.result import EvalReport


class StructuredMetric(BaseMetric):
    """Field-level structured-output metric for deepeval.

    The test case supplies the two documents: `actual_output` is evaluated
    against `expected_output`, both parsed as JSON or YAML when they are
    strings.

    What deepeval reports comes from the last `measure` call. The whole
    `EvalReport` stays on `report`, for a caller who wants the per-field detail
    deepeval has no place for.

    Attributes:
        config: The configuration each `measure` call evaluates under.
        threshold: The score a document must reach to count as a pass.
        include_reason: Whether to fill `reason` with the failure summary.
        score: The last document score; 0.0 before the first `measure`.
        success: Whether that score cleared `threshold`.
        reason: The failure summary, or `None` when reasons are off.
        report: The last full report, or `None` before the first `measure`.

    Example:
        >>> from deepeval import assert_test  # doctest: +SKIP
        >>> from deepeval.test_case import LLMTestCase
        >>> from structured_eval.integrations.deepeval import StructuredMetric
        >>> from structured_eval.models import EvalConfig
        >>> metric = StructuredMetric(EvalConfig(), threshold=0.85)
        >>> case = LLMTestCase(
        ...     input="Extract the invoice.",
        ...     actual_output='{"status": "paid"}',
        ...     expected_output='{"status": "paid"}'
        ... )
        >>> assert_test(case, [metric])  # doctest: +SKIP
        >>> metric.score, metric.reason  # doctest: +SKIP
        (1.0, 'all fields passed')
    """

    def __init__(
        self,
        config: EvalConfig | None = None,
        threshold: float = 0.5,
        *,
        include_reason: bool = True,
    ) -> None:
        """Configure the metric; nothing is evaluated until `measure`.

        Args:
            config: Field configuration, metrics and policies. Defaults to
                `EvalConfig()`, which compares every scalar with `ExactMatch`.
            threshold: The score a document must reach to count as a pass.
            include_reason: Whether to summarise the failures into `reason`.
        """
        self.config = config or EvalConfig()
        self.threshold = threshold
        self.include_reason = include_reason
        self.score: float = 0.0
        self.success: bool = False
        self.reason: str | None = None
        self.report: EvalReport | None = None

    def measure(self, test_case: Any, *args: Any, **kwargs: Any) -> float:
        """Evaluate one test case, recording the verdict on the metric.

        Args:
            test_case: A deepeval test case with `actual_output` and
                `expected_output`.
            *args: Ignored; accepted for deepeval's calling convention.
            **kwargs: Ignored; accepted for deepeval's calling convention.

        Returns:
            The document score, 0.0 when there was no ground truth to score
            against.
        """
        self.report = evaluate(
            test_case.actual_output, test_case.expected_output, self.config
        )
        score, success, reason = verdict(self.report, self.threshold)
        self.score = 0.0 if score is None else score
        self.success = success
        self.reason = reason if self.include_reason else None
        return self.score

    async def a_measure(self, test_case: Any, *args: Any, **kwargs: Any) -> float:
        """Async form of `measure`; the evaluation itself does no I/O.

        Args:
            test_case: A deepeval test case with `actual_output` and
                `expected_output`.
            *args: Forwarded to `measure`.
            **kwargs: Forwarded to `measure`.

        Returns:
            The document score, 0.0 when there was no ground truth to score
            against.
        """
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        """Whether the last `measure` call cleared the threshold."""
        return self.success

    @property
    def __name__(self) -> str:
        """The label deepeval prints for this metric."""
        return "Structured Eval"
