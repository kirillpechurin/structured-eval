"""structured_eval — field-level evaluation of structured LLM outputs.

NOTE: this package is mid-rewrite (v3, see rfcs/user-stories/technical_details_v3).
Comparison is a metric; the engine and evaluate() land in Stage 6. So far the
data structures and the field/object metric hierarchy are wired up.
"""

from structured_eval.core.config import (
    ArrayFieldConfig,
    ArrayStrategy,
    EvalConfig,
    ExtraKeysPolicy,
    FieldConfig,
    NullPolicy,
    ObjectFieldConfig,
)
from structured_eval._evaluate import evaluate
from structured_eval.core.context import EvalContext
from structured_eval.core.result import EvalReport, FieldScore, RegressionDiff, RuleResult
from structured_eval.core.sample import Sample
from structured_eval.diff.structured_diff import (
    DiffEntry,
    DiffType,
    StructuredDiff,
    structured_diff,
)
from structured_eval.metrics import (
    ArrayAccuracy,
    ArrayCardinality,
    ArrayF1,
    ArrayMetric,
    ArrayPRF1,
    ArrayPrecision,
    ArrayRecall,
    Coverage,
    ExactMatch,
    FieldMetric,
    Fuzzy,
    Metric,
    NodeMetric,
    NormalizedMatch,
    Numeric,
    ObjectAccuracy,
    ObjectF1,
    ObjectMetric,
    ObjectPRF1,
    ObjectPrecision,
    ObjectRecall,
    ObjectValidity,
    OverallScore,
    Presence,
    RootMetric,
    SchemaValidity,
    TokenF1,
    TypeMatch,
)
from structured_eval.nodes import (
    ArrayMatchResult,
    ArrayNode,
    EvalNode,
    ObjectNode,
    ScalarNode,
)
from structured_eval.rules.dsl import Rule
from structured_eval.utils.flatten import flatten

__all__ = [
    # entrypoint
    "evaluate",
    # core
    "Sample",
    "EvalContext",
    "EvalConfig",
    "FieldConfig",
    "ObjectFieldConfig",
    "ArrayFieldConfig",
    "NullPolicy",
    "ExtraKeysPolicy",
    "ArrayStrategy",
    "EvalReport",
    "FieldScore",
    "RegressionDiff",
    "RuleResult",
    # nodes
    "EvalNode",
    "ScalarNode",
    "ObjectNode",
    "ArrayNode",
    "ArrayMatchResult",
    # metrics — base hierarchy
    "Metric",
    "FieldMetric",
    "ObjectMetric",
    "ArrayMetric",
    "RootMetric",
    "NodeMetric",
    # metrics — field
    "ExactMatch",
    "NormalizedMatch",
    "Numeric",
    "TokenF1",
    "Fuzzy",
    "Presence",
    "TypeMatch",
    # metrics — object
    "ObjectAccuracy",
    "ObjectPrecision",
    "ObjectRecall",
    "ObjectF1",
    "ObjectPRF1",
    "ObjectValidity",
    # metrics — array
    "ArrayAccuracy",
    "ArrayPrecision",
    "ArrayRecall",
    "ArrayF1",
    "ArrayPRF1",
    "ArrayCardinality",
    # metrics — root
    "OverallScore",
    "SchemaValidity",
    "Coverage",
    # rules / utils / diff
    "Rule",
    "flatten",
    "structured_diff",
    "StructuredDiff",
    "DiffEntry",
    "DiffType",
]
