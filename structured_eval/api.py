from __future__ import annotations

from typing import Any

from structured_eval.engine.evaluator import Evaluator
from structured_eval.model.config import EvalConfig
from structured_eval.model.result import BatchEvalReport, ConsistencyReport, EvalReport
from structured_eval.model.sample import Sample


def _is_batch(actual: Any) -> bool:
    """A list of Samples is a batch; a bare list is a single array-root doc."""
    return isinstance(actual, list) and all(isinstance(x, Sample) for x in actual)


def evaluate(
    actual: Any,
    expected: Any = None,
    config: EvalConfig | None = None,
    *,
    source: str | None = None,
) -> EvalReport:
    """Evaluate one document against an expected reference → ``EvalReport``.

    Two call shapes:
    - ``evaluate(actual, expected, config=...)`` — shorthand for one document;
    - ``evaluate(sample, config=...)`` — one ``Sample``.

    A bare ``list`` is a single document with an array root, not a batch. To
    evaluate several samples use :func:`evaluate_batch`. Thin wrapper over
    ``Evaluator``.
    """
    if _is_batch(actual):
        raise TypeError(
            "evaluate() takes a single document; pass a list of Samples to evaluate_batch()"
        )
    sample = (
        actual
        if isinstance(actual, Sample)
        else Sample(actual=actual, expected=expected, source=source)
    )
    return Evaluator(config).evaluate_one(sample)


def evaluate_batch(
    samples: list[Sample],
    config: EvalConfig | None = None,
) -> BatchEvalReport:
    """Evaluate a list of ``Sample`` s → ``BatchEvalReport``.

    Each sample carries its own ``actual`` / ``expected`` / ``source``; the
    aggregate report exposes per-sample reports plus batch-level metrics. Thin
    wrapper over ``Evaluator``.
    """
    return Evaluator(config).evaluate_batch(samples)


def evaluate_consistency(
    runs: list[Sample],
    config: EvalConfig | None = None,
    *,
    variance_threshold: float = 0.05,
) -> ConsistencyReport:
    """Measure run-to-run stability across repeated outputs of one prompt.

    ``runs`` are several outputs for the same input (with or without a shared
    ``expected``). Fields whose score varies at most ``variance_threshold``
    across runs are reported as stable, the rest as unstable. Thin wrapper over
    ``Evaluator``.
    """
    return Evaluator(config).evaluate_consistency(runs, variance_threshold=variance_threshold)
