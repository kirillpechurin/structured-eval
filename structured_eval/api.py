"""The three public entry points of the package.

`evaluate` scores one document, `evaluate_batch` a list of samples, and
`evaluate_consistency` measures how stable repeated runs of one prompt are. All
three are thin wrappers over `Evaluator`: they normalise the call shape a caller
used and delegate. Nothing else is exported from `structured_eval` itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from structured_eval.engine.evaluator import Evaluator
from structured_eval.models.sample import Sample

if TYPE_CHECKING:
    from structured_eval.models.config import EvalConfig
    from structured_eval.models.result import (
        BatchEvalReport,
        ConsistencyReport,
        EvalReport,
    )


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
    """Evaluate one document against an expected reference.

    Two call shapes are accepted:

    - `evaluate(actual, expected, config=...)` — the two documents directly;
    - `evaluate(sample, config=...)` — one `Sample` carrying both.

    A bare `list` is one document with an array root, not a batch; a list of
    `Sample` objects belongs in `evaluate_batch` and is rejected here. Thin
    wrapper over `Evaluator`.

    Args:
        actual: The document under evaluation — a dict, a list, a scalar, a
            JSON/YAML string, or a `Sample` carrying all three arguments.
        expected: The reference document. Ignored when `actual` is a `Sample`.
        config: Field configuration, metrics and policies. Defaults to
            `EvalConfig()`, which compares every scalar with `ExactMatch`.
        source: Grounding text for the faithfulness metrics. Ignored when
            `actual` is a `Sample`.

    Returns:
        The report for this document: `score` is the root node's representative
        metric, `field_scores` holds one entry per path, `metrics` gives the
        cross-field view per metric name.

    Raises:
        TypeError: If `actual` is a list of `Sample` objects.

    Example:
        >>> from structured_eval import evaluate
        >>> report = evaluate({"status": "paid", "total": 12},
        ...                   {"status": "paid", "total": 10})
        >>> round(report.score, 2)
        0.5
        >>> float(report.field_scores["total"].score)
        0.0
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
    """Evaluate a list of samples and aggregate the results.

    Each sample carries its own `actual` / `expected` / `source`, so a batch may
    mix documents freely; only the configuration is shared. Thin wrapper over
    `Evaluator`.

    Args:
        samples: The documents to score, one `Sample` each.
        config: Field configuration, metrics and policies. Defaults to
            `EvalConfig()`, which compares every scalar with `ExactMatch`.

    Returns:
        The aggregate report: the per-sample reports, the mean of each metric
        across them, and the batch rates `perfect_response_rate` and
        `parse_error_rate`.

    Example:
        >>> from structured_eval import evaluate_batch
        >>> from structured_eval.models import Sample
        >>> report = evaluate_batch([
        ...     Sample(actual={"status": "paid"}, expected={"status": "paid"}),
        ...     Sample(actual={"status": "due"}, expected={"status": "paid"}),
        ... ])
        >>> report.perfect_response_rate
        0.5
        >>> round(report.score, 2)
        0.5
    """
    return Evaluator(config).evaluate_batch(samples)


def evaluate_consistency(
    runs: list[Sample],
    config: EvalConfig | None = None,
    *,
    variance_threshold: float = 0.05,
) -> ConsistencyReport:
    """Measure run-to-run stability across repeated outputs of one prompt.

    Stability is a property of the model, not of the reference, so the runs may
    share an `expected` or carry none at all. Thin wrapper over `Evaluator`.

    Args:
        runs: Several outputs produced for the same input, one `Sample` each.
        config: Field configuration, metrics and policies. Defaults to
            `EvalConfig()`, which compares every scalar with `ExactMatch`.
        variance_threshold: The bar a field's score variance must stay at or
            below to count as stable.

    Returns:
        The stability report: the per-run reports, each leaf field's score
        variance, the split into `stable_fields` / `unstable_fields`, and the
        mean and variance of the document score.

    Example:
        >>> from structured_eval import evaluate_consistency
        >>> from structured_eval.models import Sample
        >>> runs = [Sample(actual={"status": "paid", "total": total},
        ...                expected={"status": "paid", "total": 10})
        ...         for total in (10, 10, 12)]
        >>> report = evaluate_consistency(runs)
        >>> report.stable_fields
        ['status']
        >>> report.unstable_fields
        ['total']
    """
    return Evaluator(config).evaluate_consistency(
        runs, variance_threshold=variance_threshold
    )
