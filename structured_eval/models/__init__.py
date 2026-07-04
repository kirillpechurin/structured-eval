"""structured_eval.models — the pydantic data layer.

Single home for every user-facing data model, re-exported here so callers use
one path — ``from structured_eval.models import <X>`` — rather than reaching
into individual submodules:

- configuration: ``EvalConfig`` + the ``*FieldConfig`` family and policies;
- input: ``Sample``, ``EvalContext``;
- the ``EvalNode`` tree: ``EvalNode`` / ``ScalarNode`` / ``ObjectNode`` /
  ``ArrayNode`` (+ ``ArrayMatchResult``);
- metric values: ``MetricResult`` / ``MetricCollection``;
- reports & scores: ``EvalReport`` / ``BatchEvalReport`` / ``ConsistencyReport``,
  ``FieldScore`` / ``RuleResult`` / ``RegressionDiff`` / ``EvalWarning`` /
  ``WarningType`` / ``NodeType``.
"""

from structured_eval.models.config import (
    ArrayFieldConfig,
    ArrayStrategy,
    EvalConfig,
    ExtraKeysPolicy,
    FieldConfig,
    ObjectFieldConfig,
)
from structured_eval.models.context import EvalContext
from structured_eval.models.metric_result import MetricCollection, MetricResult
from structured_eval.models.nodes import (
    ArrayMatchResult,
    ArrayNode,
    EvalNode,
    ObjectNode,
    ScalarNode,
)
from structured_eval.models.result import (
    BatchEvalReport,
    ConsistencyReport,
    EvalReport,
    EvalWarning,
    FieldScore,
    NodeType,
    RegressionDiff,
    RuleResult,
    WarningType,
)
from structured_eval.models.sample import Sample

__all__ = [
    "ArrayFieldConfig",
    "ArrayMatchResult",
    "ArrayNode",
    "ArrayStrategy",
    "BatchEvalReport",
    "ConsistencyReport",
    "EvalConfig",
    "EvalContext",
    "EvalNode",
    "EvalReport",
    "EvalWarning",
    "ExtraKeysPolicy",
    "FieldConfig",
    "FieldScore",
    "MetricCollection",
    "MetricResult",
    "NodeType",
    "ObjectFieldConfig",
    "ObjectNode",
    "RegressionDiff",
    "RuleResult",
    "Sample",
    "ScalarNode",
    "WarningType",
]
