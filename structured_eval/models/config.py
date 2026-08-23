"""`EvalConfig` and the per-field configs that shape an evaluation.

A config states which metrics apply where, how much each field weighs and how
arrays are aligned. The field configs nest to mirror the document:

- `FieldConfig` — a scalar leaf.
- `ObjectFieldConfig` — a dict.
- `ArrayFieldConfig` — a list.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_FIELD_WEIGHT: float = 1.0


# ── Enums ─────────────────────────────────────────────────────────────────────


class ExtraKeysPolicy(StrEnum):
    """How to treat keys present in actual but absent from expected.

    Attributes:
        IGNORE: Extra keys are skipped.
        PENALIZE: Extra keys lower precision.
    """

    IGNORE = "ignore"
    PENALIZE = "penalize"


class ArrayStrategy(StrEnum):
    """How to align actual array items with expected ones.

    Attributes:
        BY_INDEX: Pair the i-th actual item with the i-th expected one.
        BY_KEY: Match on a shared unique field, named in `ArrayFieldConfig.params`.
        HUNGARIAN: Optimal one-to-one assignment by element similarity.
    """

    BY_INDEX = "by_index"
    BY_KEY = "by_key"
    HUNGARIAN = "hungarian"


# ── Field configs ───────────────────────────────────────────────────────────


class FieldConfig(BaseModel):
    """Configuration for a scalar (leaf) field.

    Attributes:
        metrics: This field's metrics, *added* to those cascading from
            `EvalConfig.metrics`.
        key_metric: Which metric is the match criterion the parent object or
            array uses — an instance or a registered name; `None` → `ExactMatch`.
        threshold: The bar `key_metric` must clear to count as a true positive.
        weight: Relative importance in the parent's weighted aggregation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metrics: list[Any] | None = None
    key_metric: Any = None
    threshold: float | None = None
    weight: float = DEFAULT_FIELD_WEIGHT


class ObjectFieldConfig(BaseModel):
    """Configuration for an object (dict) field.

    Attributes:
        fields: Per-key configuration for this object's children.
        weight: Relative importance in the parent's weighted aggregation.
        threshold: The bar the representative score must clear to count as a TP.
        metrics: This node's metrics, added to those cascading from `EvalConfig`.
        key_metric: This node's *representative* (roll-up) metric — an instance
            or a registered name; `None` → a global distributable `key_metric`,
            else `MeanScore`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    fields: dict[str, AnyFieldConfig] = Field(default_factory=dict)
    weight: float = DEFAULT_FIELD_WEIGHT
    threshold: float | None = None
    metrics: list[Any] | None = None
    key_metric: Any = None


class ArrayFieldConfig(BaseModel):
    """Configuration for an array (list) field.

    `params` carries the options of the chosen `strategy`, interpreted by the
    aligner `make_aligner` builds, so a new strategy adds no new fields here:

    - `BY_INDEX` — empty.
    - `BY_KEY` — `{"key": <field|[fields]|None>, "key_metric": <metric|name>,
      "threshold": <float>}`; several `key` fields form a composite key, scored
      as their mean.
    - `HUNGARIAN` — `{"scorer": <Scorer|dict[str, Scorer]|None>, "threshold":
      <float>, "key": <field|[fields]|None>}`; `scorer` as a per-field dict
      scores arrays of objects, and with `key` set it binds a scorer per key
      field.

    Attributes:
        item: The type and config of each element.
        strategy: Which aligner pairs actual items with expected ones.
        params: That strategy's options, as above.
        weight: Relative importance in the parent's weighted aggregation.
        threshold: The bar the representative score must clear to count as a TP.
        metrics: This node's metrics, added to those cascading from `EvalConfig`.
        key_metric: This node's *representative* (roll-up) metric. Distinct from
            a `key_metric` inside `params`, which is the element-matching metric.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    item: FieldConfig | ObjectFieldConfig | None = None
    strategy: ArrayStrategy = ArrayStrategy.BY_INDEX
    params: dict[str, Any] = Field(default_factory=dict)
    weight: float = DEFAULT_FIELD_WEIGHT
    threshold: float | None = None
    metrics: list[Any] | None = None
    key_metric: Any = None


AnyFieldConfig = FieldConfig | ObjectFieldConfig | ArrayFieldConfig


def weight_of(cfg: AnyFieldConfig | None) -> float:
    """The aggregation weight a field config contributes (``1.0`` when absent)."""
    return cfg.weight if cfg is not None else DEFAULT_FIELD_WEIGHT


# ── Eval config ───────────────────────────────────────────────────────────────


class EvalConfig(BaseModel):
    """Top-level evaluation configuration.

    `metrics` and `default_*_metrics` are different knobs:

    - `metrics` *cascades* — every listed metric is added to every node whose
      type it fits.
    - `default_*_metrics` is the *fallback* — used only by nodes that ended up
      with no metric at all, so each still has one for its `key_metric` to
      summarise.

    Attributes:
        metrics: Metric instances cascaded to every node whose type they fit.
        fields: Per-field configuration; accepts canonical nested configs and
            dot-notation keys (`"vendor.name"`) as syntactic sugar.
        root: The root node's type; inferred from `type(actual)` when omitted.
        key_metric: The metric whose value becomes `report.score`.
        extra_keys: What to do with keys absent from the expected document.
        default_scalar_metrics: Replaces the built-in `ExactMatch` fallback;
            `None` keeps it, and a list must be non-empty.
        default_object_metrics: The same for `ObjectAccuracy`.
        default_array_metrics: The same for `ArrayAccuracy`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metrics: list[Any] = Field(default_factory=list)
    fields: dict[str, AnyFieldConfig] = Field(default_factory=dict)
    root: ObjectFieldConfig | ArrayFieldConfig | None = None
    key_metric: Any = None
    extra_keys: ExtraKeysPolicy = ExtraKeysPolicy.IGNORE
    default_scalar_metrics: list[Any] | None = None
    default_object_metrics: list[Any] | None = None
    default_array_metrics: list[Any] | None = None

    @field_validator(
        "default_scalar_metrics", "default_object_metrics", "default_array_metrics"
    )
    @classmethod
    def _non_empty(cls, value: list[Any] | None, info: Any) -> list[Any] | None:
        if value is not None and not value:
            raise ValueError(
                f"{info.field_name} must list at least one metric; "
                f"omit it (None) to keep the built-in default."
            )
        return value


ObjectFieldConfig.model_rebuild()
ArrayFieldConfig.model_rebuild()
EvalConfig.model_rebuild()
