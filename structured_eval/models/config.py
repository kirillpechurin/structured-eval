from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_FIELD_WEIGHT: float = 1.0


# ── Enums ─────────────────────────────────────────────────────────────────────


class ExtraKeysPolicy(StrEnum):
    """How to treat keys present in actual but absent from expected."""

    IGNORE = "ignore"  # extra keys are skipped
    PENALIZE = "penalize"  # extra keys lower precision


class ArrayStrategy(StrEnum):
    """How to align actual array items with expected ones."""

    BY_INDEX = "by_index"  # pair the i-th with the i-th
    BY_KEY = "by_key"  # match on a shared unique field (see ArrayFieldConfig.params)
    HUNGARIAN = "hungarian"  # optimal one-to-one assignment by element similarity


# ── Field configs ───────────────────────────────────────────────────────────


class FieldConfig(BaseModel):
    """Configuration for a scalar (leaf) field.

    In v3 comparison is a metric: ``metrics`` is the field's metric list, *added*
    to the metrics cascading from ``EvalConfig.metrics``. ``key_metric`` names
    which of them is the match criterion the parent object/array uses (a metric
    instance or its registered name; ``None`` → ``ExactMatch``); ``threshold`` is
    the bar it must clear to count as a true positive.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metrics: list[Any] | None = (
        None  # list[Metric]; added to the cascading config.metrics
    )
    key_metric: Any = None  # Metric | name str used as the parent's match criterion
    threshold: float | None = None
    weight: float = DEFAULT_FIELD_WEIGHT


class ObjectFieldConfig(BaseModel):
    """Configuration for an object (dict) field.

    ``key_metric`` picks this object node's *representative* (roll-up) metric —
    a metric instance or a registered name, ``None`` → the default (a global
    distributable ``key_metric``, else ``MeanScore``). Same override
    ``FieldConfig`` offers for scalar leaves.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    fields: dict[str, AnyFieldConfig] = Field(default_factory=dict)
    weight: float = DEFAULT_FIELD_WEIGHT
    threshold: float | None = None
    metrics: list[Any] | None = None
    key_metric: Any = None  # Metric | name str: this node's representative score


class ArrayFieldConfig(BaseModel):
    """Configuration for an array (list) field.

    ``item`` describes the type and config of each element. ``strategy`` picks
    the aligner; ``params`` carries that strategy's options (interpreted by the
    aligner built in ``make_aligner``), so new strategies add no new fields here:

    * ``BY_INDEX`` → ``params`` empty.
    * ``BY_KEY`` → ``{"key": <field|None>, "key_metric": <metric|name>,
      "threshold": <float>}``. The generalized ``BY_KEY`` subsumes value- and
      similarity-based matching (technical_details_v3 §5).
    * ``HUNGARIAN`` → ``{"scorer": <Scorer | dict[str, Scorer] | None>,
      "threshold": <float>, "key": <field|None>}``. Optimal one-to-one
      assignment; ``scorer`` as a per-field dict scores arrays of objects.

    ``key_metric`` picks this array node's *representative* (roll-up) metric —
    a metric instance or a registered name, ``None`` → the default. This is
    distinct from a ``key_metric`` inside ``params`` for the ``BY_KEY``
    strategy, which is the element-alignment matching metric.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    item: FieldConfig | ObjectFieldConfig | None = None
    strategy: ArrayStrategy = ArrayStrategy.BY_INDEX
    params: dict[str, Any] = Field(default_factory=dict)  # strategy-specific options
    weight: float = DEFAULT_FIELD_WEIGHT
    threshold: float | None = None
    metrics: list[Any] | None = None
    key_metric: Any = None  # Metric | name str: this node's representative score


AnyFieldConfig = FieldConfig | ObjectFieldConfig | ArrayFieldConfig


def weight_of(cfg: AnyFieldConfig | None) -> float:
    """The aggregation weight a field config contributes (``1.0`` when absent)."""
    return cfg.weight if cfg is not None else DEFAULT_FIELD_WEIGHT


# ── Eval config ───────────────────────────────────────────────────────────────


class EvalConfig(BaseModel):
    """Top-level evaluation configuration.

    Metrics are class instances (e.g. ``ObjectF1()``, ``SchemaValidity(...)``).
    ``fields`` accepts canonical nested configs as well as dot-notation keys
    (``"vendor.name"``) as syntactic sugar. ``root`` explicitly declares the
    type of the root node; when omitted it is inferred from ``type(actual)``.

    ``metrics`` and ``default_*_metrics`` are different knobs. ``metrics``
    *cascades*: every listed metric is added to every node whose type it fits.
    ``default_*_metrics`` is the *fallback*: it replaces the built-in default
    (``ExactMatch`` / ``ObjectAccuracy`` / ``ArrayAccuracy``) for nodes of that
    type which ended up with no metric at all, so that every node still carries
    one for its ``key_metric`` to summarise. A node reached by a cascading
    global, or carrying its own ``cfg.metrics``, never sees the default. ``None``
    keeps the built-in; a list must be non-empty (a node with zero metrics would
    silently score ``0.0``).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metrics: list[Any] = Field(
        default_factory=list
    )  # list[Metric]; cascade by type to all nodes
    fields: dict[str, AnyFieldConfig] = Field(default_factory=dict)
    root: ObjectFieldConfig | ArrayFieldConfig | None = None
    key_metric: Any = None  # Metric whose value becomes report.score
    extra_keys: ExtraKeysPolicy = ExtraKeysPolicy.IGNORE
    # list[Metric] | name str; None → the engine's built-in default for the type
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
