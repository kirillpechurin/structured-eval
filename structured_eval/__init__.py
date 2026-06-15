"""structured_eval — field-level evaluation of structured LLM outputs.

Layered architecture (dependencies point downward):

- ``model``      — pure data: ``Sample``, ``EvalConfig``, ``EvalContext``,
                   ``nodes``, ``EvalReport`` (and friends);
- ``metrics``    — every metric as its own module/package (field/object/array/
                   root), incl. ``SchemaValidity``, ``RulePassRate``,
                   ``Faithfulness``; ``alignment`` / ``formats`` / ``utils``;
- ``engine``     — ``Evaluator`` plus the phase classes (parse → build tree →
                   run metrics → build report) and batch aggregation;
- ``reporting``  — console rendering; ``integrations`` — host-framework adapters.

``evaluate`` / ``evaluate_consistency`` are thin wrappers over ``engine.Evaluator``.
"""

from structured_eval.api import evaluate, evaluate_consistency
from structured_eval.metrics import (
    ArrayAccuracy,
    ArrayCardinality,
    ArrayF1,
    ArrayMetric,
    ArrayPrecision,
    ArrayPRF1,
    ArrayRecall,
    Coverage,
    ExactMatch,
    Faithfulness,
    FieldMetric,
    Fuzzy,
    Metric,
    NodeMetric,
    NormalizedMatch,
    Numeric,
    ObjectAccuracy,
    ObjectF1,
    ObjectMetric,
    ObjectPrecision,
    ObjectPRF1,
    ObjectRecall,
    ObjectValidity,
    OverallScore,
    Presence,
    RootMetric,
    Rule,
    RulePassRate,
    SchemaValidity,
    TokenF1,
    TypeMatch,
)
from structured_eval.model.config import (
    ArrayFieldConfig,
    ArrayStrategy,
    EvalConfig,
    ExtraKeysPolicy,
    FieldConfig,
    NullPolicy,
    ObjectFieldConfig,
)
from structured_eval.model.context import EvalContext
from structured_eval.model.nodes import (
    ArrayMatchResult,
    ArrayNode,
    EvalNode,
    ObjectNode,
    ScalarNode,
)
from structured_eval.model.result import (
    BatchEvalReport,
    ConsistencyReport,
    EvalReport,
    FieldScore,
    RegressionDiff,
    RuleResult,
)
from structured_eval.model.sample import Sample
from structured_eval.utils.flatten import flatten
from structured_eval.utils.structured_diff import (
    DiffEntry,
    DiffType,
    StructuredDiff,
    structured_diff,
)

__all__ = [
    # entrypoint
    "evaluate",
    "evaluate_consistency",
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
    "BatchEvalReport",
    "ConsistencyReport",
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
    "Faithfulness",
    "RulePassRate",
    # rules / utils / diff
    "Rule",
    "flatten",
    "structured_diff",
    "StructuredDiff",
    "DiffEntry",
    "DiffType",
]
