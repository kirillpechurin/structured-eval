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
) -> EvalReport | BatchEvalReport:
    """Evaluate one document, or a batch, against expected references.

    Three call shapes:
    - ``evaluate(actual, expected, config=...)`` — shorthand for one document;
    - ``evaluate(sample, config=...)`` — one ``Sample``;
    - ``evaluate([Sample(...), ...], config=...)`` — a batch → ``BatchEvalReport``.

    A bare ``list`` passed as ``actual`` is a single document with an array root,
    not a batch; wrap documents in ``Sample`` to evaluate a batch. Thin wrapper
    over ``Evaluator``.
    """
    if _is_batch(actual):
        # a batch has no `expected`; tolerate evaluate(samples, cfg) positionally
        if config is None and isinstance(expected, EvalConfig):
            config = expected
        return Evaluator(config).evaluate_batch(actual)

    sample = (
        actual
        if isinstance(actual, Sample)
        else Sample(actual=actual, expected=expected, source=source)
    )
    return Evaluator(config).evaluate_one(sample)


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
    return Evaluator(config).evaluate_consistency(
        runs, variance_threshold=variance_threshold
    )
