from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_NUMERIC_TOLERANCE: float = 0.01
DEFAULT_FUZZY_THRESHOLD: float = 0.8
DEFAULT_FIELD_WEIGHT: float = 1.0


# ── Enums ─────────────────────────────────────────────────────────────────────


class NullHandling(StrEnum):
    LENIENT = "lenient"  # null and missing treated as equivalent
    STRICT = "strict"  # null != missing


class ExtraKeys(StrEnum):
    IGNORE = "ignore"
    PENALIZE = "penalize"
    REWARD = "reward"


class ArrayStrategy(StrEnum):
    BY_INDEX = "by_index"  # pair i-th with i-th (MVP)
    BY_KEY = "by_key"  # match on a shared unique field (v0.2)
    BEST_MATCH = "best_match"  # Hungarian algorithm (v0.2)


class NumericMode(StrEnum):
    RELATIVE = "relative"  # |a - e| / |e| < tolerance
    ABSOLUTE = "absolute"  # |a - e| < tolerance


# ── Matcher types ─────────────────────────────────────────────────────────────
#
# Internal frozen dataclasses — lightweight value objects used as tags.
# Public API goes through the MatchMode namespace below.


@dataclass(frozen=True)
class _Exact:
    """Exact equality: actual == expected."""

    def __str__(self) -> str:
        return "exact"


@dataclass(frozen=True)
class _Normalized:
    """Case-insensitive, whitespace-normalized equality."""

    def __str__(self) -> str:
        return "normalized"


@dataclass(frozen=True)
class _Numeric:
    """Numeric comparison within a tolerance band."""

    tolerance: float = DEFAULT_NUMERIC_TOLERANCE
    mode: NumericMode = NumericMode.RELATIVE

    def __str__(self) -> str:
        return f"numeric(tolerance={self.tolerance}, mode={self.mode})"


@dataclass(frozen=True)
class _TokenF1:
    """Token-level F1 score after normalization. Default for long strings."""

    def __str__(self) -> str:
        return "token_f1"


@dataclass(frozen=True)
class _Fuzzy:
    """Fuzzy string similarity via RapidFuzz."""

    threshold: float = DEFAULT_FUZZY_THRESHOLD

    def __str__(self) -> str:
        return f"fuzzy(threshold={self.threshold})"


@dataclass(frozen=True)
class _Custom:
    """User-supplied comparison function."""

    fn: Callable[[Any, Any], bool | float]
    name: str = ""

    def __str__(self) -> str:
        return f"custom:{self.name}" if self.name else "custom"


MatcherType = _Exact | _Normalized | _Numeric | _TokenF1 | _Fuzzy | _Custom


class MatchMode:
    """Namespace for matcher constructors.

    Simple matchers are class-level singletons; parametrized matchers
    are factory methods.

    Examples:
        MatchMode.EXACT
        MatchMode.NORMALIZED
        MatchMode.NUMERIC(tolerance=0.05, mode=NumericMode.RELATIVE)
        MatchMode.CUSTOM(lambda a, e: a.lower() in e.lower(), name="substr")
    """

    EXACT: ClassVar[_Exact] = _Exact()
    NORMALIZED: ClassVar[_Normalized] = _Normalized()
    TOKEN_F1: ClassVar[_TokenF1] = _TokenF1()
    FUZZY: ClassVar[_Fuzzy] = _Fuzzy()

    @staticmethod
    def NUMERIC(
        tolerance: float = DEFAULT_NUMERIC_TOLERANCE,
        mode: NumericMode = NumericMode.RELATIVE,
    ) -> _Numeric:
        """Numeric matcher with configurable tolerance.

        Args:
            tolerance: Maximum allowed deviation. For RELATIVE mode, expressed
                as a fraction (0.01 = 1%). For ABSOLUTE mode, in the same units
                as the values being compared.
            mode: RELATIVE computes |a - e| / |e|; ABSOLUTE computes |a - e|.
        """
        return _Numeric(tolerance=tolerance, mode=mode)

    @staticmethod
    def CUSTOM(
        fn: Callable[[Any, Any], bool | float],
        name: str = "",
    ) -> _Custom:
        """Wrap an arbitrary comparison function.

        Args:
            fn: Callable(actual, expected) -> bool or float in [0.0, 1.0].
            name: Human-readable name shown in reports.
        """
        return _Custom(fn=fn, name=name)


# ── Field config ──────────────────────────────────────────────────────────────


class FieldConfig(BaseModel):
    """Per-field evaluation configuration.

    Attach to a field name in EvalConfig.fields to override any global default
    for that specific field. Supports nested configuration for arrays of objects
    via item_fields.

    Example:
        FieldConfig(
            matcher=MatchMode.NUMERIC(tolerance=0.01),
            weight=3.0,
            required=True,
        )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    matcher: MatcherType | None = Field(
        default=None,
        description="Matcher for this field. Falls back to EvalConfig.default_matcher when None.",
    )
    weight: float = Field(
        default=DEFAULT_FIELD_WEIGHT,
        gt=0,
        description="Relative weight of this field in the aggregated score.",
    )
    required: bool = Field(
        default=False,
        description="If True, a missing field is treated as a hard failure regardless of null_handling.",
    )
    null_handling: NullHandling | None = Field(
        default=None,
        description="Override global null_handling for this field. None = use EvalConfig.null_handling.",
    )
    extra_keys: ExtraKeys | None = Field(
        default=None,
        description="Override global extra_keys for this field. None = use EvalConfig.extra_keys.",
    )
    array_strategy: ArrayStrategy = Field(
        default=ArrayStrategy.BY_INDEX,
        description="How to align array items with expected.",
    )
    array_key: str | None = Field(
        default=None,
        description="Field name used as the unique key when array_strategy=BY_KEY.",
    )
    derived: bool = Field(
        default=False,
        description=(
            "Mark field as derived (computed from other fields). "
            "Derived fields are excluded from hallucination scoring."
        ),
    )
    item_fields: dict[str, FieldConfig] | None = Field(
        default=None,
        description="Per-field configuration applied recursively to each item in an array of objects.",
    )


FieldConfig.model_rebuild()


# ── Eval config ───────────────────────────────────────────────────────────────


class EvalConfig(BaseModel):
    """Top-level evaluation configuration passed to evaluate().

    Provides global defaults that individual FieldConfig entries can override.
    All parameters have sensible defaults so the config is optional for simple
    use cases.

    Example:
        EvalConfig(
            default_matcher=MatchMode.NORMALIZED,
            null_handling=NullHandling.LENIENT,
            fields={
                "id": FieldConfig(matcher=MatchMode.EXACT, required=True),
                "total": FieldConfig(matcher=MatchMode.NUMERIC(tolerance=0.01), weight=3.0),
            },
        )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    default_matcher: MatcherType = Field(
        default_factory=lambda: MatchMode.TOKEN_F1,
        description="Matcher applied to all fields without an explicit FieldConfig.",
    )
    null_handling: NullHandling = Field(
        default=NullHandling.LENIENT,
        description="How to handle null vs missing field comparisons.",
    )
    extra_keys: ExtraKeys = Field(
        default=ExtraKeys.IGNORE,
        description="How to handle keys present in actual but absent from expected.",
    )
    numeric_tolerance: float = Field(
        default=DEFAULT_NUMERIC_TOLERANCE,
        ge=0,
        description="Default relative tolerance for NUMERIC matchers without an explicit tolerance.",
    )
    fields: dict[str, FieldConfig] | None = Field(
        default=None,
        description="Per-field overrides. Keys are field names or JSONPath expressions.",
    )
    rules: list[Any] = Field(
        default_factory=list,
        description="Rule objects to evaluate against the document. Populated via the rules module.",
    )
    json_schema: Any | None = Field(
        default=None,
        description="Pydantic model class or JSON Schema dict used for structural validation.",
    )
