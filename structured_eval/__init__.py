"""structured_eval — field-level evaluation of structured LLM outputs.

NOTE: this package is mid-rewrite (v2). evaluate(), matchers/ and metrics/
land in Stages 4–6; only the data structures are wired up so far.
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
from structured_eval.core.context import EvalContext
from structured_eval.core.result import EvalReport, FieldScore, RegressionDiff, RuleResult
from structured_eval.core.sample import Sample
from structured_eval.diff.structured_diff import (
    DiffEntry,
    DiffType,
    StructuredDiff,
    structured_diff,
)
from structured_eval.matchers import (
    ExactMatcher,
    FuzzyMatcher,
    JaccardMatcher,
    MatcherBase,
    NumericMatcher,
    RegexNormalizedMatcher,
    TokenF1Matcher,
    UrlMatcher,
)
from structured_eval.nodes import (
    ArrayMatchResult,
    ArrayNode,
    EvalNode,
    FieldPair,
    ObjectNode,
    ScalarNode,
)
from structured_eval.rules.dsl import Rule
from structured_eval.utils.flatten import flatten

__all__ = [
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
    "FieldPair",
    "ArrayMatchResult",
    # matchers
    "MatcherBase",
    "ExactMatcher",
    "RegexNormalizedMatcher",
    "NumericMatcher",
    "TokenF1Matcher",
    "JaccardMatcher",
    "FuzzyMatcher",
    "UrlMatcher",
    # rules / utils / diff
    "Rule",
    "flatten",
    "structured_diff",
    "StructuredDiff",
    "DiffEntry",
    "DiffType",
]
