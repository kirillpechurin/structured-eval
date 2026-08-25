"""Framework-agnostic mapping from an EvalReport to a (score, success, reason).

Shared by every integration so the host-specific classes stay thin. Tested
directly, without any host library installed.
"""

from __future__ import annotations

from structured_eval.models import EvalReport, NodeType

_MAX_REASONS = 5


def reason_text(report: EvalReport) -> str:
    """Human-readable explanation of a report, focused on what failed.

    A host framework shows one line per assertion, so the text names at most
    five failed fields and counts the rest. Scalars are quoted as
    `actual != expected`; object and array nodes carry a score instead, since
    "these two objects differ" says nothing a number does not.

    Args:
        report: The report to explain.

    Returns:
        The parse error, `"all fields passed"`, or the failed fields joined
        into one line.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.integrations import reason_text
        >>> reason_text(evaluate({"status": "paid"}, {"status": "paid"}))
        'all fields passed'
        >>> report = evaluate({"status": "paid", "total": 12},
        ...                   {"status": "due", "total": 12})
        >>> reason_text(report)
        "2 field(s) failed: $: score 0.5; status: 'paid' != 'due'"
    """
    if report.parse_error:
        return f"parse error: {report.parse_error_message or 'could not parse output'}"

    failed = report.failed_fields()
    if not failed:
        return "all fields passed"

    parts = []
    for fs in list(failed.values())[:_MAX_REASONS]:
        if fs.node_type == NodeType.SCALAR:
            parts.append(f"{fs.path}: {fs.actual!r} != {fs.expected!r}")
        else:
            parts.append(
                f"{fs.path}: score {fs.score:.2g}" if fs.score is not None else fs.path
            )
    if len(failed) > _MAX_REASONS:
        parts.append(f"... +{len(failed) - _MAX_REASONS} more")

    head = f"{len(failed)} field(s) failed: "
    return head + "; ".join(parts)


def verdict(report: EvalReport, threshold: float) -> tuple[float | None, bool, str]:
    """Reduce a report to `(score, success, reason)` for a host framework.

    Every host wants the same three things out of an evaluation, so each adapter
    reduces the report here rather than inventing its own rule. A score of `None`
    leaves the pass/fail bar inapplicable, which counts as a failure.

    Args:
        report: The report to reduce.
        threshold: The score a document must reach to count as a pass.

    Returns:
        The key-metric value (`None` when there is no key metric or no ground
        truth), whether the document passed, and the explanation.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.integrations import verdict
        >>> report = evaluate({"status": "paid", "total": 12},
        ...                   {"status": "due", "total": 12})
        >>> score, success, _reason = verdict(report, 0.8)
        >>> score, success
        (0.5, False)
    """
    score = report.score
    success = not report.parse_error and score is not None and score >= threshold
    return score, success, reason_text(report)
