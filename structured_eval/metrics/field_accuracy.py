from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from structured_eval.core.config import (
    DEFAULT_FIELD_WEIGHT,
    ArrayStrategy,
    EvalConfig,
    ExtraKeys,
    FieldConfig,
    MatcherType,
    NullHandling,
)
from structured_eval.core.result import FieldScore, FieldStatus
from structured_eval.metrics.matchers import match

# Sentinel that distinguishes "key absent from dict" from "key present with value None".
_MISSING: Any = object()


# ── Public API ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FieldAccuracyResult:
    """Aggregated output of a single field-level comparison."""

    field_scores: dict[str, FieldScore]
    f1: float
    precision: float
    recall: float
    type_error_rate: float


def evaluate_fields(
    actual: dict[str, Any],
    expected: dict[str, Any] | list[dict[str, Any]],
    config: EvalConfig | None = None,
) -> FieldAccuracyResult:
    """Evaluate field-level accuracy of actual against expected.

    When expected is a list of reference dicts, each is evaluated independently
    and the result with the highest F1 is returned (best-match multi-reference).
    """
    cfg = config or EvalConfig()

    if isinstance(expected, list):
        if not expected:
            raise ValueError("expected list must not be empty")
        candidates = [_evaluate_single(actual, ref, cfg) for ref in expected]
        return max(candidates, key=lambda r: r.f1)

    return _evaluate_single(actual, expected, cfg)


# ── Single-reference evaluation ───────────────────────────────────────────────


def _evaluate_single(
    actual: dict[str, Any],
    expected: dict[str, Any],
    config: EvalConfig,
) -> FieldAccuracyResult:
    field_scores = _compare_dict(actual, expected, config.fields, config)
    f1, precision, recall = _aggregate_f1(field_scores, config.fields, config.extra_keys)
    return FieldAccuracyResult(
        field_scores=field_scores,
        f1=f1,
        precision=precision,
        recall=recall,
        type_error_rate=_type_error_rate(field_scores),
    )


# ── Dict comparison ───────────────────────────────────────────────────────────


def _compare_dict(
    actual: dict[str, Any],
    expected: dict[str, Any],
    field_configs: dict[str, FieldConfig] | None,
    eval_config: EvalConfig,
) -> dict[str, FieldScore]:
    scores: dict[str, FieldScore] = {}
    for key in set(actual) | set(expected):
        fc = (field_configs or {}).get(key)
        scores[key] = _compare_field(
            key,
            actual.get(key, _MISSING),
            expected.get(key, _MISSING),
            fc,
            eval_config,
        )
    return scores


# ── Field comparison ──────────────────────────────────────────────────────────


def _compare_field(
    key: str,
    actual_val: Any,
    expected_val: Any,
    field_config: FieldConfig | None,
    eval_config: EvalConfig,
) -> FieldScore:
    null_handling = _eff_null_handling(field_config, eval_config)

    # Determine "absence" according to null_handling policy.
    # LENIENT: None and _MISSING are equivalent (both count as absent).
    # STRICT:  only _MISSING is absent; None is an explicit value.
    if null_handling == NullHandling.LENIENT:
        actual_absent = actual_val is _MISSING or actual_val is None
        expected_absent = expected_val is _MISSING or expected_val is None
    else:
        actual_absent = actual_val is _MISSING
        expected_absent = expected_val is _MISSING

    # Normalise _MISSING → None for storage in FieldScore.
    stored_actual = None if actual_val is _MISSING else actual_val
    stored_expected = None if expected_val is _MISSING else expected_val

    # ── Both sides absent ─────────────────────────────────────────────────────
    if actual_absent and expected_absent:
        # In STRICT mode, two explicit None values are an exact match.
        if actual_val is None and expected_val is None:
            return FieldScore(
                key=key, expected=None, actual=None,
                score=1.0, matcher="exact", status=FieldStatus.MATCH,
            )
        return FieldScore(
            key=key, expected=stored_expected, actual=stored_actual,
            score=1.0, matcher="null_match", status=FieldStatus.NULL_MATCH,
        )

    # ── Expected absent, actual present → EXTRA ───────────────────────────────
    if expected_absent:
        score = _extra_score(_eff_extra_keys(field_config, eval_config))
        return FieldScore(
            key=key, expected=stored_expected, actual=stored_actual,
            score=score, matcher="extra", status=FieldStatus.EXTRA,
        )

    # ── Actual absent, expected present → MISS ────────────────────────────────
    if actual_absent:
        return FieldScore(
            key=key, expected=stored_expected, actual=stored_actual,
            score=0.0, matcher="miss", status=FieldStatus.MISS,
        )

    # ── Both present in STRICT mode but one side is None ─────────────────────
    # (only reachable in STRICT: None is a real value, not "absent")
    if actual_val is None:
        return FieldScore(
            key=key, expected=stored_expected, actual=None,
            score=0.0, matcher="miss", status=FieldStatus.MISS,
        )
    if expected_val is None:
        score = _extra_score(_eff_extra_keys(field_config, eval_config))
        return FieldScore(
            key=key, expected=None, actual=stored_actual,
            score=score, matcher="extra", status=FieldStatus.EXTRA,
        )

    # ── Both sides carry non-None values ─────────────────────────────────────
    if isinstance(actual_val, dict) and isinstance(expected_val, dict):
        return _compare_nested_object(key, actual_val, expected_val, field_config, eval_config)

    if isinstance(actual_val, list) and isinstance(expected_val, list):
        strategy = field_config.array_strategy if field_config else ArrayStrategy.BY_INDEX
        if strategy == ArrayStrategy.BY_INDEX:
            return _compare_array_by_index(key, actual_val, expected_val, field_config, eval_config)
        # TODO: BY_KEY and BEST_MATCH are planned for array_metrics.py (v0.2)
        raise NotImplementedError(
            f"ArrayStrategy.{strategy} is not yet supported (planned for v0.2)"
        )

    matcher = _eff_matcher(field_config, eval_config, key, expected_val)
    score = match(matcher, actual_val, expected_val)
    return FieldScore(
        key=key, expected=stored_expected, actual=stored_actual,
        score=score, matcher=str(matcher), status=_status(score),
    )


# ── Nested object ─────────────────────────────────────────────────────────────


def _compare_nested_object(
    key: str,
    actual: dict[str, Any],
    expected: dict[str, Any],
    field_config: FieldConfig | None,
    eval_config: EvalConfig,
) -> FieldScore:
    # item_fields carries per-child configs when the parent is an object inside an array;
    # for plain nested objects there are no per-child overrides at this level.
    child_configs = field_config.item_fields if field_config else None
    children = _compare_dict(actual, expected, child_configs, eval_config)
    child_f1, _, _ = _aggregate_f1(children, child_configs, eval_config.extra_keys)
    return FieldScore(
        key=key, expected=expected, actual=actual,
        score=child_f1, matcher="object_f1", status=_status(child_f1),
        children=children,
    )


# ── Array BY_INDEX ────────────────────────────────────────────────────────────


def _compare_array_by_index(
    key: str,
    actual: list[Any],
    expected: list[Any],
    field_config: FieldConfig | None,
    eval_config: EvalConfig,
) -> FieldScore:
    if not actual and not expected:
        return FieldScore(
            key=key, expected=expected, actual=actual,
            score=1.0, matcher="array_by_index", status=FieldStatus.MATCH,
        )

    child_configs = field_config.item_fields if field_config else None
    children: dict[str, FieldScore] = {}

    for i in range(max(len(actual), len(expected))):
        a_item = actual[i] if i < len(actual) else _MISSING
        e_item = expected[i] if i < len(expected) else _MISSING
        idx = str(i)

        if isinstance(a_item, dict) and isinstance(e_item, dict):
            children[idx] = _compare_nested_object(idx, a_item, e_item, None, eval_config)
        else:
            fc_i = (child_configs or {}).get(idx) if child_configs else None
            children[idx] = _compare_field(idx, a_item, e_item, fc_i, eval_config)

    arr_score = sum(fs.score for fs in children.values()) / len(children)
    return FieldScore(
        key=key, expected=expected, actual=actual,
        score=arr_score, matcher="array_by_index", status=_status(arr_score),
        children=children,
    )


# ── F1 aggregation ────────────────────────────────────────────────────────────


def _aggregate_f1(
    field_scores: dict[str, FieldScore],
    field_configs: dict[str, FieldConfig] | None,
    extra_keys: ExtraKeys,
) -> tuple[float, float, float]:
    def w(k: str) -> float:
        fc = (field_configs or {}).get(k)
        return fc.weight if fc else DEFAULT_FIELD_WEIGHT

    # Recall: fields that were expected (everything except pure EXTRA).
    recall_keys = [k for k, fs in field_scores.items() if fs.status != FieldStatus.EXTRA]

    # Precision: fields that were produced by actual.
    # IGNORE: extra fields are excluded from the denominator (not penalised).
    # PENALIZE / REWARD: extra fields are included.
    if extra_keys == ExtraKeys.IGNORE:
        precision_keys = [
            k for k, fs in field_scores.items()
            if fs.status not in (FieldStatus.MISS, FieldStatus.EXTRA)
        ]
    else:
        precision_keys = [k for k, fs in field_scores.items() if fs.status != FieldStatus.MISS]

    recall_w = sum(w(k) for k in recall_keys)
    precision_w = sum(w(k) for k in precision_keys)

    recall = (
        sum(w(k) * field_scores[k].score for k in recall_keys) / recall_w
        if recall_w else 1.0
    )
    precision = (
        sum(w(k) * field_scores[k].score for k in precision_keys) / precision_w
        if precision_w else 1.0
    )

    denom = precision + recall
    f1 = 2 * precision * recall / denom if denom else 0.0
    return f1, precision, recall


# ── Type error rate ───────────────────────────────────────────────────────────


def _type_error_rate(field_scores: dict[str, FieldScore]) -> float:
    """Fraction of compared fields where type(actual) != type(expected)."""
    comparable = [
        fs for fs in field_scores.values()
        if fs.actual is not None
        and fs.expected is not None
        and fs.status not in (FieldStatus.MISS, FieldStatus.EXTRA, FieldStatus.NULL_MATCH)
    ]
    if not comparable:
        return 0.0
    errors = sum(1 for fs in comparable if type(fs.actual) is not type(fs.expected))
    return errors / len(comparable)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _eff_null_handling(fc: FieldConfig | None, ec: EvalConfig) -> NullHandling:
    return fc.null_handling if fc and fc.null_handling is not None else ec.null_handling


def _eff_extra_keys(fc: FieldConfig | None, ec: EvalConfig) -> ExtraKeys:
    return fc.extra_keys if fc and fc.extra_keys is not None else ec.extra_keys


def _eff_matcher(
    fc: FieldConfig | None, ec: EvalConfig, key: str, expected_val: Any
) -> MatcherType:
    if fc and fc.matcher is not None:
        return fc.matcher
    return ec.default_matcher


def _extra_score(extra_keys: ExtraKeys) -> float:
    return 0.5 if extra_keys == ExtraKeys.REWARD else 0.0


def _status(score: float) -> FieldStatus:
    """Map a matcher score to a status for fields where both sides are present.

    MISS is reserved for truly absent fields and must be set explicitly in
    null/missing handling above — never via this helper.
    """
    if score >= 1.0:
        return FieldStatus.MATCH
    return FieldStatus.PARTIAL
