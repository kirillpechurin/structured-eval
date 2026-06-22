"""Framework-agnostic mapping from an EvalReport to a (score, success, reason).

Shared by every integration so the host-specific classes stay thin. Tested
directly, without any host library installed.
"""

from __future__ import annotations

from structured_eval.model.result import EvalReport

_MAX_REASONS = 5


def reason_text(report: EvalReport) -> str:
    """Human-readable explanation of a report, focused on what failed."""
    if report.parse_error:
        return f"parse error: {report.parse_error_message or 'could not parse output'}"

    failed = report.failed_fields()
    if not failed:
        return "all fields passed"

    parts = []
    for fs in list(failed.values())[:_MAX_REASONS]:
        if fs.node_type == "scalar":
            parts.append(f"{fs.path}: {fs.actual!r} != {fs.expected!r}")
        else:
            parts.append(f"{fs.path}: score {fs.score:.2g}" if fs.score is not None else fs.path)
    if len(failed) > _MAX_REASONS:
        parts.append(f"... +{len(failed) - _MAX_REASONS} more")

    head = f"{len(failed)} field(s) failed: "
    return head + "; ".join(parts)


def verdict(report: EvalReport, threshold: float) -> tuple[float | None, bool, str]:
    """Reduce a report to (score, success, reason) for a host framework.

    ``score`` is the key-metric value (``None`` when no key metric / no ground
    truth). ``success`` requires a parsed document and ``score >= threshold``;
    when ``score`` is ``None`` the pass/fail bar cannot be applied → ``False``.
    """
    score = report.score
    success = not report.parse_error and score is not None and score >= threshold
    return score, success, reason_text(report)
