"""`Evaluator` — the orchestrator that runs the phases for one config."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from structured_eval.engine.aggregator import BatchAggregator
from structured_eval.engine.metric_runner import MetricRunner
from structured_eval.engine.parser import Parser
from structured_eval.engine.report_builder import ReportBuilder
from structured_eval.engine.tree_builder import TreeBuilder
from structured_eval.models.config import EvalConfig
from structured_eval.models.context import EvalContext
from structured_eval.models.result import BatchEvalReport, ConsistencyReport, EvalReport
from structured_eval.utils.flatten import flatten

if TYPE_CHECKING:
    from structured_eval.models.sample import Sample


class Evaluator:
    """Orchestrates the three evaluation phases for one config.

    Holds the `EvalConfig` and the phase collaborators (parse → build tree →
    run metrics → build report), and aggregates batches.

    The package-level `evaluate` / `evaluate_batch` / `evaluate_consistency`
    functions are thin wrappers over this; construct an `Evaluator` directly to
    score many samples under one config without re-resolving it each time.

    Attributes:
        config: The configuration every phase reads.

    Example:
        >>> from structured_eval.engine import Evaluator
        >>> from structured_eval.models import EvalConfig, Sample
        >>> evaluator = Evaluator(EvalConfig())
        >>> report = evaluator.evaluate_one(
        ...     Sample(actual={"status": "paid"}, expected={"status": "due"})
        ... )
        >>> report.score
        0.0
        >>> batch = evaluator.evaluate_batch(
        ...     [Sample(actual={"status": "paid"}, expected={"status": "paid"})]
        ... )
        >>> batch.perfect_response_rate
        1.0
    """

    def __init__(self, config: EvalConfig | None = None):
        """Store the config and build the phase collaborators.

        Args:
            config: Field configuration, metrics and policies. Defaults to
                `EvalConfig()`, which compares every scalar with `ExactMatch`.
        """
        self.config = config or EvalConfig()
        self._parser = Parser()
        self._runner = MetricRunner()
        self._report_builder = ReportBuilder()
        self._aggregator = BatchAggregator()

    def evaluate_one(self, sample: Sample) -> EvalReport:
        """Evaluate a single document against its expected reference.

        Args:
            sample: The document to score, with its reference and source.

        Returns:
            The report for this document, or a report with `parse_error` set
            when either side could not be parsed.

        Example:
            >>> from structured_eval.engine import Evaluator
            >>> from structured_eval.models import Sample
            >>> report = Evaluator().evaluate_one(
            ...     Sample(actual={"status": "paid"}, expected={"status": "due"})
            ... )
            >>> report.score
            0.0
            >>> report.field_scores["status"].expected
            'due'
        """
        actual, actual_err = self._parser.parse(sample.actual)
        expected, expected_err = self._parser.parse(sample.expected)
        error = actual_err or expected_err
        if error is not None:
            return EvalReport(parse_error=True, parse_error_message=error)

        context = EvalContext(
            actual=actual,
            expected=expected,
            source=sample.source,
            flat_actual=_flat(actual),
            flat_expected=_flat(expected),
            config=self.config,
        )

        root, warnings = TreeBuilder(
            context
        ).build()  # phase 1: structure + per-node metrics
        self._runner.run(
            root
        )  # phase 2: compute post-order, each node's key_metric last
        return self._report_builder.build(root, context, warnings)  # phase 3

    def evaluate_batch(self, samples: list[Sample]) -> BatchEvalReport:
        """Evaluate a list of documents and aggregate the results.

        Args:
            samples: The documents to score, one `Sample` each.

        Returns:
            The aggregate report over the per-sample reports.

        Example:
            >>> from structured_eval.engine import Evaluator
            >>> from structured_eval.models import Sample
            >>> batch = Evaluator().evaluate_batch([
            ...     Sample(actual={"status": "paid"}, expected={"status": "paid"}),
            ...     Sample(actual={"status": "due"}, expected={"status": "paid"}),
            ... ])
            >>> batch.perfect_response_rate, round(batch.score, 2)
            (0.5, 0.5)
        """
        return self._aggregator.batch([self.evaluate_one(s) for s in samples])

    def evaluate_consistency(
        self, runs: list[Sample], *, variance_threshold: float = 0.05
    ) -> ConsistencyReport:
        """Measure run-to-run stability across repeated outputs of one prompt.

        Args:
            runs: Several outputs produced for the same input, one `Sample`
                each.
            variance_threshold: The bar a field's score variance must stay at
                or below to count as stable.

        Returns:
            The stability report over the per-run reports.

        Example:
            >>> from structured_eval.engine import Evaluator
            >>> from structured_eval.models import Sample
            >>> runs = [Sample(actual={"status": status},
            ...                expected={"status": "paid"})
            ...         for status in ("paid", "paid", "due")]
            >>> Evaluator().evaluate_consistency(runs).unstable_fields
            ['status']
        """
        reports = [self.evaluate_one(s) for s in runs]
        return self._aggregator.consistency(reports, variance_threshold)


def _flat(data: Any) -> dict[str, Any]:
    return flatten(data) if isinstance(data, (dict, list)) else {}
