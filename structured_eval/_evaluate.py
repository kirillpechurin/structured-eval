from __future__ import annotations

from typing import Any

from structured_eval.core.config import EvalConfig
from structured_eval.core.context import EvalContext
from structured_eval.core.result import EvalReport
from structured_eval.core.sample import Sample
from structured_eval.engine.match import build_tree
from structured_eval.engine.parser import parse
from structured_eval.engine.report_builder import build_report
from structured_eval.engine.compute import run
from structured_eval.utils.flatten import flatten


def evaluate(
    actual: Any,
    expected: Any = None,
    config: EvalConfig | None = None,
    *,
    source: str | None = None,
) -> EvalReport:
    """Evaluate one document against an expected reference.

    Accepts either a ``Sample`` (``evaluate(sample, config=...)``) or the
    shorthand ``evaluate(actual, expected, config=...)``. Batch evaluation
    (``list[Sample]``) lands in Stage 9.
    """
    if isinstance(actual, Sample):
        sample = actual
    else:
        sample = Sample(actual=actual, expected=expected, source=source)

    return _evaluate_single(sample, config or EvalConfig())


def _flat(data: Any) -> dict:
    return flatten(data) if isinstance(data, (dict, list)) else {}


def _evaluate_single(sample: Sample, config: EvalConfig) -> EvalReport:
    actual, actual_err = parse(sample.actual)
    expected, expected_err = parse(sample.expected)
    error = actual_err or expected_err
    if error is not None:
        return EvalReport(parse_error=True, parse_error_message=error)

    context = EvalContext(
        actual=actual,
        expected=expected,
        source=sample.source,
        flat_actual=_flat(actual),
        flat_expected=_flat(expected),
        config=config,
    )

    metrics = list(config.metrics)
    if config.key_metric is not None:  # ensure the score metric is computed
        metrics.append(config.key_metric)

    root, warnings = build_tree(context)  # phase 1: structure + leaf metrics
    run(root, metrics)  # phase 2: object/array/root metrics
    return build_report(root, context, warnings)  # phase 3: flatten
