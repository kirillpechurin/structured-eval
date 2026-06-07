from structured_eval.core.config import EvalConfig, FieldConfig, MatchMode
from structured_eval.core.evaluator import evaluate
from structured_eval.core.result import EvalReport, FieldScore, RuleResult
from structured_eval.diff.structured_diff import DiffEntry, DiffType, StructuredDiff, structured_diff
from structured_eval.rules.dsl import Rule

__all__ = [
    "evaluate",
    "EvalConfig",
    "FieldConfig",
    "MatchMode",
    "EvalReport",
    "FieldScore",
    "Rule",
    "RuleResult",
    "structured_diff",
    "StructuredDiff",
    "DiffEntry",
    "DiffType",
]
