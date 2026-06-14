from __future__ import annotations

from typing import Any

from structured_eval.core.config import EvalConfig
from structured_eval.core.context import EvalContext
from structured_eval.core.result import BatchEvalReport, ConsistencyReport, EvalReport
from structured_eval.core.sample import Sample
from structured_eval.engine.aggregate import batch_report, consistency_report
from structured_eval.engine.match import build_tree
from structured_eval.engine.parser import parse
from structured_eval.engine.report_builder import build_report
from structured_eval.engine.compute import run
from structured_eval.utils.flatten import flatten


def _is_batch(actual: Any) -> bool:
    """A list of Samples is a batch; a bare list is a single array-root doc."""
    return isinstance(actual, list) and all(isinstance(x, Sample) for x in actual)


def evaluate(
    actual: Any,
    expected: Any = None,
    config: EvalConfig | None = None,
    *,
    source: str | None = None,
) -> EvalReport | BatchEvalReport:
    """Evaluate one document, or a batch, against expected references.

    Three call shapes:
    - ``evaluate(actual, expected, config=...)`` — shorthand for one document;
    - ``evaluate(sample, config=...)`` — one ``Sample``;
    - ``evaluate([Sample(...), ...], config=...)`` — a batch → ``BatchEvalReport``.

    A bare ``list`` passed as ``actual`` is a single document with an array root,
    not a batch; wrap documents in ``Sample`` to evaluate a batch.
    """
    if _is_batch(actual):
        # a batch has no `expected`; tolerate evaluate(samples, cfg) positionally
        if config is None and isinstance(expected, EvalConfig):
            config = expected
        cfg = config or EvalConfig()
        return batch_report([_evaluate_single(s, cfg) for s in actual])

    cfg = config or EvalConfig()

    sample = actual if isinstance(actual, Sample) else Sample(
        actual=actual, expected=expected, source=source
    )
    return _evaluate_single(sample, cfg)


def evaluate_consistency(
    runs: list[Sample],
    config: EvalConfig | None = None,
    *,
    variance_threshold: float = 0.05,
) -> ConsistencyReport:
    """Measure run-to-run stability across repeated outputs of one prompt.

    ``runs`` are several outputs for the same input (with or without a shared
    ``expected``). Fields whose score varies at most ``variance_threshold``
    across runs are reported as stable, the rest as unstable.
    """
    cfg = config or EvalConfig()
    reports = [_evaluate_single(s, cfg) for s in runs]
    return consistency_report(reports, variance_threshold)


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
