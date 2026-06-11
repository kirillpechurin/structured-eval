from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_FIELD_WEIGHT: float = 1.0


# ── Enums ─────────────────────────────────────────────────────────────────────


class NullPolicy(StrEnum):
    """How to treat a field whose actual or expected value is null."""

    IGNORE = "ignore"  # null/null is skipped, does not affect the score
    PENALIZE = "penalize"  # null in place of a value → score 0.0 (default)
    REQUIRE_MATCH = "require_match"  # null == null → 1.0; null ≠ value → 0.0


class ExtraKeysPolicy(StrEnum):
    """How to treat keys present in actual but absent from expected."""

    IGNORE = "ignore"  # extra keys are skipped
    PENALIZE = "penalize"  # extra keys lower precision


class ArrayStrategy(StrEnum):
    """How to align actual array items with expected ones."""

    BY_INDEX = "by_index"  # pair the i-th with the i-th
    BY_KEY = "by_key"  # match on a shared unique field (see ArrayFieldConfig.key)


# ── Field configs ───────────────────────────────────────────────────────────


@dataclass
class FieldConfig:
    """Configuration for a scalar (leaf) field.

    ``matcher=None`` means AUTO — the matcher is auto-detected from the key
    name and value type. ``metrics`` overrides the global metric list for
    this field only.
    """

    matcher: Any = None  # Matcher instance, or None for AUTO
    weight: float = DEFAULT_FIELD_WEIGHT
    required: bool = False
    null_policy: NullPolicy | None = None  # None → inherit EvalConfig.null_policy
    threshold: float | None = None
    exclude: bool = False
    derived: bool = False  # excluded from Faithfulness
    metrics: list[Any] | None = None  # list[Metric], overrides global


@dataclass
class ObjectFieldConfig:
    """Configuration for an object (dict) field."""

    fields: dict[str, FieldConfig | ObjectFieldConfig | ArrayFieldConfig] = field(
        default_factory=dict
    )
    weight: float = DEFAULT_FIELD_WEIGHT
    threshold: float | None = None
    exclude: bool = False
    metrics: list[Any] | None = None


@dataclass
class ArrayFieldConfig:
    """Configuration for an array (list) field.

    ``item`` describes the type and config of each element. ``key`` names the
    unique field used to align elements when ``strategy=BY_KEY``.
    """

    item: FieldConfig | ObjectFieldConfig | None = None
    strategy: ArrayStrategy = ArrayStrategy.BY_INDEX
    key: str | None = None  # required for BY_KEY
    weight: float = DEFAULT_FIELD_WEIGHT
    threshold: float | None = None
    exclude: bool = False
    metrics: list[Any] | None = None


AnyFieldConfig = FieldConfig | ObjectFieldConfig | ArrayFieldConfig


# ── Eval config ───────────────────────────────────────────────────────────────


@dataclass
class EvalConfig:
    """Top-level evaluation configuration.

    Metrics are class instances (e.g. ``ObjectF1()``, ``SchemaValidity(...)``).
    ``fields`` accepts canonical nested configs as well as dot-notation keys
    (``"vendor.name"``) as syntactic sugar. ``root`` explicitly declares the
    type of the root node; when omitted it is inferred from ``type(actual)``.
    """

    metrics: list[Any] = field(default_factory=list)  # list[Metric]
    fields: dict[str, AnyFieldConfig] = field(default_factory=dict)
    root: ObjectFieldConfig | ArrayFieldConfig | None = None
    default_matcher: Any = None  # Matcher applied to fields without an explicit one
    key_metric: Any = None  # Metric whose value becomes report.score
    null_policy: NullPolicy = NullPolicy.PENALIZE
    extra_keys: ExtraKeysPolicy = ExtraKeysPolicy.IGNORE
