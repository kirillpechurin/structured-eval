"""JSON round-trip for EvalReport (dataclass ↔ plain dict)."""

from __future__ import annotations

from structured_eval.core.config import ArrayStrategy
from structured_eval.core.result import EvalReport, FieldScore, RuleResult
from structured_eval.nodes.array_node import ArrayMatchResult


def _field_score_to_dict(fs: FieldScore) -> dict:
    return {
        "path": fs.path,
        "node_type": fs.node_type,
        "actual": fs.actual,
        "expected": fs.expected,
        "metrics": dict(fs.metrics),
        "score": fs.score,
        "threshold": fs.threshold,
    }


def _array_match_to_dict(m: ArrayMatchResult) -> dict:
    return {
        "strategy": str(m.strategy),
        "matched": [list(pair) for pair in m.matched],
        "missed": list(m.missed),
        "spurious": list(m.spurious),
    }


def report_to_dict(r: EvalReport) -> dict:
    """Serialize an EvalReport to a JSON-friendly dict."""
    return {
        "score": r.score,
        "score_label": r.score_label,
        "metrics": dict(r.metrics),
        "field_scores": {p: _field_score_to_dict(fs) for p, fs in r.field_scores.items()},
        "array_matches": {p: _array_match_to_dict(m) for p, m in r.array_matches.items()},
        "rule_results": [
            {"name": rr.name, "passed": rr.passed, "message": rr.message}
            for rr in r.rule_results
        ],
        "parse_error": r.parse_error,
        "parse_error_message": r.parse_error_message,
        "schema_errors": list(r.schema_errors),
        "warnings": list(r.warnings),
    }


def report_from_dict(d: dict) -> EvalReport:
    """Reconstruct an EvalReport from a dict produced by ``report_to_dict``."""
    field_scores = {
        p: FieldScore(
            path=fs["path"],
            node_type=fs["node_type"],
            actual=fs.get("actual"),
            expected=fs.get("expected"),
            metrics=dict(fs.get("metrics", {})),
            score=fs.get("score"),
            threshold=fs.get("threshold"),
        )
        for p, fs in d.get("field_scores", {}).items()
    }
    array_matches = {
        p: ArrayMatchResult(
            strategy=ArrayStrategy(m["strategy"]),
            matched=[tuple(pair) for pair in m.get("matched", [])],
            missed=list(m.get("missed", [])),
            spurious=list(m.get("spurious", [])),
        )
        for p, m in d.get("array_matches", {}).items()
    }
    rule_results = [
        RuleResult(name=rr["name"], passed=rr["passed"], message=rr.get("message", ""))
        for rr in d.get("rule_results", [])
    ]
    return EvalReport(
        score=d.get("score"),
        score_label=d.get("score_label"),
        metrics=dict(d.get("metrics", {})),
        field_scores=field_scores,
        array_matches=array_matches,
        rule_results=rule_results,
        parse_error=d.get("parse_error", False),
        parse_error_message=d.get("parse_error_message"),
        schema_errors=list(d.get("schema_errors", [])),
        warnings=list(d.get("warnings", [])),
    )
