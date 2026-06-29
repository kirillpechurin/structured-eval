from __future__ import annotations

from typing import TYPE_CHECKING, Any

from structured_eval.engine.aggregator import BatchAggregator
from structured_eval.engine.metric_runner import MetricRunner
from structured_eval.engine.parser import Parser
from structured_eval.engine.report_builder import ReportBuilder
from structured_eval.engine.tree_builder import TreeBuilder
from structured_eval.model.config import EvalConfig
from structured_eval.model.context import EvalContext
from structured_eval.model.result import BatchEvalReport, ConsistencyReport, EvalReport
from structured_eval.utils.flatten import flatten

if TYPE_CHECKING:
    from structured_eval.model.sample import Sample


class Evaluator:
    """Orchestrates the three evaluation phases for one config.

    Holds the ``EvalConfig`` and the phase collaborators (parse → build tree →
    run metrics → build report) and aggregates batches. The module-level
    ``evaluate`` / ``evaluate_consistency`` functions are thin wrappers over this.
    """

    def __init__(self, config: EvalConfig | None = None):
        self.config = config or EvalConfig()
        self._parser = Parser()
        self._runner = MetricRunner()
        self._report_builder = ReportBuilder()
        self._aggregator = BatchAggregator()

    def evaluate_one(self, sample: Sample) -> EvalReport:
        """Evaluate a single document against its expected reference."""
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

        root, warnings = TreeBuilder(context).build()  # phase 1: structure + per-node metrics
        self._runner.run(root)  # phase 2: compute post-order, each node's key_metric last
        return self._report_builder.build(root, context, warnings)  # phase 3

    def evaluate_batch(self, samples: list[Sample]) -> BatchEvalReport:
        """Evaluate a list of documents and aggregate the results."""
        return self._aggregator.batch([self.evaluate_one(s) for s in samples])

    def evaluate_consistency(
        self, runs: list[Sample], *, variance_threshold: float = 0.05
    ) -> ConsistencyReport:
        """Measure run-to-run stability across repeated outputs of one prompt."""
        reports = [self.evaluate_one(s) for s in runs]
        return self._aggregator.consistency(reports, variance_threshold)


def _flat(data: Any) -> dict[str, Any]:
    return flatten(data) if isinstance(data, (dict, list)) else {}
