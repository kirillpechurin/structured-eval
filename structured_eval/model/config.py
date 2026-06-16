from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

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


class FieldConfig(BaseModel):
    """Configuration for a scalar (leaf) field.

    In v3 comparison is a metric: ``metrics`` is the field's metric list
    (``None`` → ``EvalConfig.default_metrics``). ``key_metric`` names which of
    them is the match criterion the parent object/array uses (a metric instance
    or its registered name; ``None`` → ``ExactMatch``); ``threshold`` is the bar
    it must clear to count as a true positive.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metrics: list[Any] | None = None  # list[Metric]; None → EvalConfig.default_metrics
    key_metric: Any = None  # Metric | name str used as the parent's match criterion
    threshold: float | None = None
    weight: float = DEFAULT_FIELD_WEIGHT
    required: bool = False
    null_policy: NullPolicy | None = None  # None → inherit EvalConfig.null_policy
    exclude: bool = False
    derived: bool = False  # excluded from Faithfulness


class ObjectFieldConfig(BaseModel):
    """Configuration for an object (dict) field."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    fields: dict[str, AnyFieldConfig] = Field(default_factory=dict)
    weight: float = DEFAULT_FIELD_WEIGHT
    threshold: float | None = None
    exclude: bool = False
    metrics: list[Any] | None = None


class ArrayFieldConfig(BaseModel):
    """Configuration for an array (list) field.

    ``item`` describes the type and config of each element. ``strategy`` picks
    the aligner; ``params`` carries that strategy's options (interpreted by the
    aligner built in ``make_aligner``), so new strategies add no new fields here:

    * ``BY_INDEX`` → ``params`` empty.
    * ``BY_KEY`` → ``{"key": <field|None>, "key_metric": <metric|name>,
      "threshold": <float>}``. The generalized ``BY_KEY`` subsumes value- and
      similarity-based matching (technical_details_v3 §5).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    item: FieldConfig | ObjectFieldConfig | None = None
    strategy: ArrayStrategy = ArrayStrategy.BY_INDEX
    params: dict[str, Any] = Field(default_factory=dict)  # strategy-specific options
    weight: float = DEFAULT_FIELD_WEIGHT
    threshold: float | None = None
    exclude: bool = False
    metrics: list[Any] | None = None


AnyFieldConfig = FieldConfig | ObjectFieldConfig | ArrayFieldConfig


# ── Eval config ───────────────────────────────────────────────────────────────


class EvalConfig(BaseModel):
    """Top-level evaluation configuration.

    Metrics are class instances (e.g. ``ObjectF1()``, ``SchemaValidity(...)``).
    ``fields`` accepts canonical nested configs as well as dot-notation keys
    (``"vendor.name"``) as syntactic sugar. ``root`` explicitly declares the
    type of the root node; when omitted it is inferred from ``type(actual)``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metrics: list[Any] = Field(default_factory=list)  # list[Metric]
    fields: dict[str, AnyFieldConfig] = Field(default_factory=dict)
    root: ObjectFieldConfig | ArrayFieldConfig | None = None
    default_metrics: list[Any] | None = None  # for fields without explicit metrics
    key_metric: Any = None  # Metric whose value becomes report.score
    null_policy: NullPolicy = NullPolicy.PENALIZE
    extra_keys: ExtraKeysPolicy = ExtraKeysPolicy.IGNORE


ObjectFieldConfig.model_rebuild()
ArrayFieldConfig.model_rebuild()
EvalConfig.model_rebuild()
