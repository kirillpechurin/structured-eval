"""The pydantic data layer — configuration, input, tree nodes and reports.

Single home for every user-facing data model, re-exported here so callers use
one path (`from structured_eval.models import ...`) rather than reaching into
the individual submodules:

- configuration — `EvalConfig` and the `*FieldConfig` family with their policies.
- input — `Sample`, `EvalContext`.
- tree nodes — `EvalNode`, `ScalarNode`, `ObjectNode`, `ArrayNode`.
- metric values — `MetricResult`, `MetricCollection`, `JudgeVerdict`.
- reports — `EvalReport`, `BatchEvalReport`, `ConsistencyReport`, `FieldScore`.
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
from structured_eval.models.metrics import (
    FieldJudgeVerdict,
    JudgeVerdict,
    MetricCollection,
    MetricResult,
)
from structured_eval.models.nodes import (
    ArrayMatchResult,
    ArrayNode,
    EvalNode,
    NodeType,
    ObjectNode,
    ScalarNode,
)
from structured_eval.models.result import (
    BatchEvalReport,
    ConsistencyReport,
    EvalReport,
    EvalWarning,
    FieldScore,
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
    "FieldJudgeVerdict",
    "FieldScore",
    "JudgeVerdict",
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
