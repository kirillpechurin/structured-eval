"""Plain-text console rendering for the report types (no hard dependency).

``ConsoleRenderer().render(report)`` returns a string; the module-level
``render`` is a thin convenience over it, used by ``EvalReport.print_summary``.
The layout is pure stdlib so it works out of the box; Rich can be layered on
for colour later.
"""

from __future__ import annotations

from typing import Any

_RULE = "─"
_BAR = "━"
_WIDTH = 60


class ConsoleRenderer:
    """Renders ``EvalReport`` / ``BatchEvalReport`` / ``ConsistencyReport``."""

    def render(self, report: Any) -> str:
        kind = type(report).__name__
        if kind == "BatchEvalReport":
            return self._render_batch(report)
        if kind == "ConsistencyReport":
            return self._render_consistency(report)
        return self._render_eval(report)

    # ── EvalReport ──────────────────────────────────────────────────────────

    def _render_eval(self, report: Any) -> str:
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

        # Document-level metrics: those a metric produced at the root ("$").
        doc_metrics = {
            name: coll.root() for name, coll in report.metrics.items() if coll.root() is not None
        }
        grid = self._metric_grid(doc_metrics, skip=report.score_label)
        if grid:
            out += ["", *grid]
        out.append(bar)

        # scalar leaves with a key-metric score
        rows = []
        for fs in report.field_scores.values():
            if fs.score is None:
                continue
            metric_name = next((k for k, v in fs.metrics.items() if v == fs.score), "score")
            rows.append(
                [
                    fs.path,
                    metric_name,
                    self._num(fs.score),
                    self._num(fs.threshold),
                    self._mark(fs.score, fs.threshold),
                ]
            )
        if rows:
            out += self._table(
                ["Field", "Metric", "Score", "Threshold", "Mark"],
                rows,
                ["<", "<", ">", ">", "^"],
            )
            out.append(bar)

        return "\n".join(out)

    # ── BatchEvalReport ───────────────────────────────────────────────────

    def _render_batch(self, report: Any) -> str:
        bar = _BAR * _WIDTH
        n = len(report.per_sample)
        out = [bar, f"  BATCH   {n} samples"]
        if report.score is not None:
            out.append(f"  mean {report.score_label or 'score'}   {report.score:.2f}")
        out.append(f"  perfect_response_rate   {report.perfect_response_rate:.2f}")
        out.append(f"  parse_error_rate        {report.parse_error_rate:.2f}")
        grid = self._metric_grid(report.metrics, skip=report.score_label)
        if grid:
            out += ["", *grid]
        out.append(bar)

        bd = report.field_breakdown()
        ranked = sorted(bd.items(), key=lambda kv: kv[1]["fail_rate"], reverse=True)
        if ranked:
            rows = [
                [path, self._num(s["mean"]), self._num(s["p95"]), self._num(s["fail_rate"])]
                for path, s in ranked
            ]
            out.append("  Field breakdown (worst first)")
            out += self._table(["Field", "mean", "p95", "fail_rate"], rows, ["<", ">", ">", ">"])
            out.append(bar)
        return "\n".join(out)

    # ── ConsistencyReport ─────────────────────────────────────────────────

    def _render_consistency(self, report: Any) -> str:
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
            out += self._table(["Field", "variance"], rows, ["<", ">"])
            out.append(bar)
        return "\n".join(out)

    # ── formatting helpers ─────────────────────────────────────────────────

    @staticmethod
    def _num(value: float | None) -> str:
        return "—" if value is None else f"{value:.2f}"

    @staticmethod
    def _mark(score: float | None, bar: float | None) -> str:
        if score is None or bar is None:
            return " "
        return "✓" if score >= bar else "✗"

    @staticmethod
    def _table(
        headers: list[str], rows: list[list[str]], aligns: list[str] | None = None
    ) -> list[str]:
        """Render a simple monospace table as a list of lines."""
        cols = list(zip(*([headers] + rows))) if rows else [[h] for h in headers]
        widths = [max(len(c) for c in col) for col in cols]
        aligns = aligns or ["<"] * len(headers)

        def fmt(cells: list[str]) -> str:
            return "  ".join(f"{c:{a}{w}}" for c, w, a in zip(cells, widths, aligns))

        lines = [fmt(headers), fmt([_RULE * w for w in widths])]
        lines += [fmt(r) for r in rows]
        return ["  " + line for line in lines]

    @staticmethod
    def _metric_grid(metrics: dict[str, float], skip: str | None = None) -> list[str]:
        """Two-per-line key/value grid of document metrics."""
        items = [(k, v) for k, v in metrics.items() if k != skip]
        if not items:
            return []
        width = max(len(k) for k, _ in items)
        cells = [f"{k:<{width}} {ConsoleRenderer._num(v)}" for k, v in items]
        lines = []
        for i in range(0, len(cells), 2):
            lines.append("  " + "     ".join(cells[i : i + 2]))
        return lines


def render(report: Any) -> str:
    """Render any of the report types to a printable string."""
    return ConsoleRenderer().render(report)
