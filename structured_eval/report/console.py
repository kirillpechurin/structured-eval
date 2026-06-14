"""Plain-text console rendering for the report types (no hard dependency).

``render(report)`` returns a string; ``print_summary()`` prints it. Rich is not
required — if it is installed it can be layered on for colour later, but the
layout here is pure stdlib so ``print_summary`` works out of the box.
"""

from __future__ import annotations

from typing import Any

_RULE = "─"
_BAR = "━"
_WIDTH = 60


def _num(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _mark(score: float | None, bar: float | None) -> str:
    if score is None or bar is None:
        return " "
    return "✓" if score >= bar else "✗"


def _table(headers: list[str], rows: list[list[str]], aligns: list[str] | None = None) -> list[str]:
    """Render a simple monospace table as a list of lines."""
    cols = list(zip(*([headers] + rows))) if rows else [[h] for h in headers]
    widths = [max(len(c) for c in col) for col in cols]
    aligns = aligns or ["<"] * len(headers)

    def fmt(cells: list[str]) -> str:
        return "  ".join(f"{c:{a}{w}}" for c, w, a in zip(cells, widths, aligns))

    lines = [fmt(headers), fmt([_RULE * w for w in widths])]
    lines += [fmt(r) for r in rows]
    return ["  " + line for line in lines]


def _metric_grid(metrics: dict[str, float], skip: str | None = None) -> list[str]:
    """Two-per-line key/value grid of document metrics."""
    items = [(k, v) for k, v in metrics.items() if k != skip]
    if not items:
        return []
    width = max(len(k) for k, _ in items)
    cells = [f"{k:<{width}} {_num(v)}" for k, v in items]
    lines = []
    for i in range(0, len(cells), 2):
        lines.append("  " + "     ".join(cells[i : i + 2]))
    return lines


# ── EvalReport ────────────────────────────────────────────────────────────────


def _render_eval(report: Any) -> str:
    out: list[str] = [_BAR * _WIDTH]

    if report.parse_error:
        out.append(f"  PARSE ERROR: {report.parse_error_message}")
        out.append(_BAR * _WIDTH)
        return "\n".join(out)

    bar = _BAR * _WIDTH
    if report.score is not None:
        verdict = "✓ PASS" if report.score >= 1.0 else "✗ FAIL"
        label = report.score_label or ""
        out.append(f"  OVERALL   {report.score:.2f}   {verdict}        {label}")
    else:
        out.append("  OVERALL   —   (no ground truth)")

    grid = _metric_grid(report.metrics, skip=report.score_label)
    if grid:
        out += ["", *grid]
    out.append(bar)

    # scalar leaves with a key-metric score
    rows = []
    for fs in report.field_scores.values():
        if fs.score is None:
            continue
        metric_name = next((k for k, v in fs.metrics.items() if v == fs.score), "score")
        rows.append([fs.path, metric_name, _num(fs.score), _num(fs.threshold), _mark(fs.score, fs.threshold)])
    if rows:
        out += _table(["Field", "Metric", "Score", "Bar", ""], rows, ["<", "<", ">", ">", "^"])
        out.append(bar)

    # object/array aggregate metrics per path
    struct = []
    for fs in report.field_scores.values():
        if fs.node_type == "scalar" or not fs.metrics or fs.path in ("$", ""):
            continue  # root metrics already shown in the header
        summary = "  ".join(f"{k} {_num(v)}" for k, v in fs.metrics.items())
        struct.append([fs.path, summary])
    if struct:
        out.append("  Structure")
        out += _table(["Path", "Metrics"], struct)
        out.append(bar)

    # failures detail
    failures = report.failed_fields()
    if failures:
        out.append("  Failures")
        for fs in failures:
            if fs.node_type == "scalar":
                out.append(f"  ✗ {fs.path}   {fs.actual!r}  →  {fs.expected!r}")
            else:
                m = report.array_matches.get(fs.path)
                if m is not None:
                    out.append(
                        f"  ✗ {fs.path}   P {m.precision:.2f}  R {m.recall:.2f}  "
                        f"({len(m.missed)} missing, {len(m.spurious)} spurious)"
                    )
                else:
                    out.append(f"  ✗ {fs.path}   score {_num(fs.score)}")
        out.append(bar)

    if report.schema_errors:
        out.append("  ⚠ schema errors")
        out += [f"  [SCHEMA] {e}" for e in report.schema_errors]
        out.append(bar)

    if report.warnings:
        out.append("  ⚠ warnings")
        out += [f"  {w}" for w in report.warnings]
        out.append(bar)

    return "\n".join(out)


# ── BatchEvalReport ─────────────────────────────────────────────────────────


def _render_batch(report: Any) -> str:
    bar = _BAR * _WIDTH
    n = len(report.per_sample)
    out = [bar, f"  BATCH   {n} samples"]
    if report.score is not None:
        out.append(f"  mean {report.score_label or 'score'}   {report.score:.2f}")
    out.append(f"  perfect_response_rate   {report.perfect_response_rate:.2f}")
    out.append(f"  parse_error_rate        {report.parse_error_rate:.2f}")
    grid = _metric_grid(report.metrics, skip=report.score_label)
    if grid:
        out += ["", *grid]
    out.append(bar)

    bd = report.field_breakdown()
    ranked = sorted(bd.items(), key=lambda kv: kv[1]["fail_rate"], reverse=True)
    if ranked:
        rows = [
            [path, _num(s["mean"]), _num(s["p95"]), _num(s["fail_rate"])]
            for path, s in ranked
        ]
        out.append("  Field breakdown (worst first)")
        out += _table(["Field", "mean", "p95", "fail_rate"], rows, ["<", ">", ">", ">"])
        out.append(bar)
    return "\n".join(out)


# ── ConsistencyReport ───────────────────────────────────────────────────────


def _render_consistency(report: Any) -> str:
    bar = _BAR * _WIDTH
    out = [bar, f"  CONSISTENCY   {len(report.per_run)} runs"]
    if report.mean_score is not None:
        out.append(f"  mean score       {report.mean_score:.2f}")
    if report.score_variance is not None:
        out.append(f"  score variance   {report.score_variance:.4f}")
    out += [
        f"  stable           {', '.join(report.stable_fields) or '—'}",
        f"  unstable         {', '.join(report.unstable_fields) or '—'}",
        bar,
    ]
    ranked = sorted(report.field_variance.items(), key=lambda kv: kv[1], reverse=True)
    if ranked:
        rows = [[path, f"{var:.4f}"] for path, var in ranked]
        out += _table(["Field", "variance"], rows, ["<", ">"])
        out.append(bar)
    return "\n".join(out)


# ── dispatch ──────────────────────────────────────────────────────────────────


def render(report: Any) -> str:
    """Render any of the report types to a printable string."""
    kind = type(report).__name__
    if kind == "BatchEvalReport":
        return _render_batch(report)
    if kind == "ConsistencyReport":
        return _render_consistency(report)
    return _render_eval(report)
