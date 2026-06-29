"""structured_eval — field-level evaluation of structured LLM outputs.

Layered architecture (dependencies point downward):

- ``model``      — pure data: ``Sample``, ``EvalConfig``, ``EvalContext``,
                   ``nodes``, ``EvalReport`` (and friends);
- ``metrics``    — every metric as its own module/package (field/object/array/
                   root), incl. ``SchemaValidity``, ``RulePassRate``,
                   ``FieldFaithfulness``; ``alignment`` / ``formats`` / ``utils``;
- ``engine``     — ``Evaluator`` plus the phase classes (parse → build tree →
                   run metrics → build report) and batch aggregation;
- ``reporting``  — console rendering; ``integrations`` — host-framework adapters.

``evaluate`` / ``evaluate_batch`` / ``evaluate_consistency`` are thin wrappers over ``engine.Evaluator``.
"""

from structured_eval.api import evaluate, evaluate_batch, evaluate_consistency
from structured_eval.metrics import (
    ArrayAccuracy,
    ArrayCardinality,
    ArrayF1,
    ArrayMetric,
    ArrayPrecision,
    ArrayPRF1,
    ArrayRecall,
    BaseMetric,
    CoverageLeafScore,
    ExactMatch,
    FieldFaithfulness,
    FieldMetric,
    Fuzzy,
    GenericMetric,
    Levenshtein,
    MeanScore,
    Metric,
    Numeric,
    NumericCloseness,
    ObjectAccuracy,
    ObjectF1,
    ObjectMetric,
    ObjectPrecision,
    ObjectPRF1,
    ObjectRecall,
    ObjectTypeValidity,
    OverallLeafScore,
    Presence,
    RegexMatch,
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
from structured_eval.model.metric_result import MetricCollection, MetricResult
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
    EvalWarning,
    FieldScore,
    RegressionDiff,
    RuleResult,
    WarningType,
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
    # metrics — array
    "ArrayAccuracy",
    "ArrayCardinality",
    "ArrayF1",
    "ArrayFieldConfig",
    "ArrayMatchResult",
    "ArrayMetric",
    "ArrayNode",
    "ArrayPRF1",
    "ArrayPrecision",
    "ArrayRecall",
    "ArrayStrategy",
    # metrics — base hierarchy
    "BaseMetric",
    "BatchEvalReport",
    "ConsistencyReport",
    "CoverageLeafScore",
    "DiffEntry",
    "DiffType",
    "EvalConfig",
    "EvalContext",
    # nodes
    "EvalNode",
    "EvalReport",
    "EvalWarning",
    # metrics — field
    "ExactMatch",
    "ExtraKeysPolicy",
    "FieldConfig",
    "FieldFaithfulness",
    "FieldMetric",
    "FieldScore",
    "Fuzzy",
    "GenericMetric",
    "Levenshtein",
    "MeanScore",
    "Metric",
    "MetricCollection",
    "MetricResult",
    "NullPolicy",
    "Numeric",
    "NumericCloseness",
    # metrics — object
    "ObjectAccuracy",
    "ObjectF1",
    "ObjectFieldConfig",
    "ObjectMetric",
    "ObjectNode",
    "ObjectPRF1",
    "ObjectPrecision",
    "ObjectRecall",
    "ObjectTypeValidity",
    # metrics — root
    "OverallLeafScore",
    "Presence",
    "RegexMatch",
    "RegressionDiff",
    "RootMetric",
    # rules / utils / diff
    "Rule",
    "RulePassRate",
    "RuleResult",
    # core
    "Sample",
    "ScalarNode",
    "SchemaValidity",
    "StructuredDiff",
    "TokenF1",
    "TypeMatch",
    "WarningType",
    # entrypoint
    "evaluate",
    "evaluate_batch",
    "evaluate_consistency",
    "flatten",
    "structured_diff",
]
