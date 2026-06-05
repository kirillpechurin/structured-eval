from __future__ import annotations

from typing import Any, Literal, overload

from structured_eval.core.config import EvalConfig
from structured_eval.core.result import EvalReport, RuleResult
from structured_eval.metrics.field_accuracy import FieldAccuracyResult, evaluate_fields
from structured_eval.parsers.base import ParseError
from structured_eval.parsers.json_parser import JsonParser
from structured_eval.schema.coverage import coverage_score as _compute_coverage
from structured_eval.schema.coverage import path_precision as _compute_path_precision
from structured_eval.schema.coverage import path_recall as _compute_path_recall
from structured_eval.schema.validator import validate as _schema_validate

_JSON_PARSER = JsonParser()


# ── Public API ────────────────────────────────────────────────────────────────


@overload
def evaluate(
    actual: str | dict[str, Any],
    expected: str | dict[str, Any] | list[dict[str, Any]],
    *,
    config: EvalConfig | None = ...,
    detailed: Literal[False] = ...,
) -> float: ...


@overload
def evaluate(
    actual: str | dict[str, Any],
    expected: str | dict[str, Any] | list[dict[str, Any]],
    *,
    config: EvalConfig | None = ...,
    detailed: Literal[True],
) -> EvalReport: ...


def evaluate(
    actual: str | dict[str, Any],
    expected: str | dict[str, Any] | list[dict[str, Any]],
    *,
    config: EvalConfig | None = None,
    detailed: bool = False,
) -> float | EvalReport:
    """Evaluate LLM structured output against expected ground truth.

    Levels of use:

        # Level 0 — just the score
        score = evaluate(actual, expected)

        # Level 1 — full report
        report = evaluate(actual, expected, detailed=True)

        # Level 2 — global config
        report = evaluate(actual, expected, config=EvalConfig(...), detailed=True)

        # Level 3 — per-field overrides (via EvalConfig.fields)
        report = evaluate(
            actual, expected,
            config=EvalConfig(fields={"id": FieldConfig(matcher=MatchMode.EXACT)}),
            detailed=True,
        )

    Args:
        actual: LLM output as a dict or JSON/YAML string.
        expected: Ground truth as a dict, string, or list of dicts.
                  List = multiple acceptable references; best F1 is returned.
        config: Evaluation configuration. Sensible defaults are used when None.
        detailed: Return a full EvalReport instead of a bare float.

    Returns:
        float when detailed=False, EvalReport when detailed=True.

    Raises:
        ParseError: If a string input cannot be parsed as JSON or YAML.
        TypeError: If a parsed input is not a dict.
    """
    cfg = config or EvalConfig()
    warnings: list[str] = []

    actual_dict = _parse(actual, warnings)
    expected_parsed = _parse_expected(expected, warnings)

    fa = evaluate_fields(actual_dict, expected_parsed, cfg)

    if not detailed:
        return fa.f1

    return _build_report(actual_dict, expected_parsed, fa, cfg, warnings)


# ── Report assembly ───────────────────────────────────────────────────────────


def _build_report(
    actual: dict[str, Any],
    expected: dict[str, Any] | list[dict[str, Any]],
    fa: FieldAccuracyResult,
    cfg: EvalConfig,
    warnings: list[str],
) -> EvalReport:
    schema_valid = None
    coverage = None
    p_recall = None
    p_precision = None

    if cfg.json_schema is not None:
        result = _schema_validate(actual, cfg.json_schema)
        schema_valid = result.valid
        coverage = _compute_coverage(actual, cfg.json_schema)

    # Path metrics require a single reference dict.
    if isinstance(expected, dict):
        p_recall = _compute_path_recall(actual, expected)
        p_precision = _compute_path_precision(actual, expected)
    # For multi-reference lists we don't know which reference was selected,
    # so path metrics are omitted.

    rule_results: list[RuleResult] = []
    rule_pass_rate = None
    if cfg.rules:
        warnings.append(
            "Rules engine is not yet implemented (planned for v0.2). Rules were skipped."
        )

    return EvalReport(
        f1=fa.f1,
        precision=fa.precision,
        recall=fa.recall,
        schema_valid=schema_valid,
        coverage_score=coverage,
        path_recall=p_recall,
        path_precision=p_precision,
        type_error_rate=fa.type_error_rate,
        rule_pass_rate=rule_pass_rate,
        rule_results=rule_results,
        field_scores=fa.field_scores,
        config=cfg,
        warnings=warnings,
    )


# ── Parsing helpers ───────────────────────────────────────────────────────────


def _parse(value: str | dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value

    try:
        result = _JSON_PARSER.parse(value)
    except ParseError as json_err:
        # YAML fallback — optional dependency
        try:
            from structured_eval.parsers.yaml_parser import YamlParser

            result = YamlParser().parse(value)
            warnings.append("Input was not valid JSON; parsed as YAML.")
        except ImportError:
            raise json_err
        except ParseError:
            raise json_err

    if not isinstance(result, dict):
        raise TypeError(f"Parsed input must be a dict, got {type(result).__name__!r}")
    return result


def _parse_expected(
    value: str | dict[str, Any] | list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any] | list[dict[str, Any]]:
    if isinstance(value, list):
        return [_parse(item, warnings) if isinstance(item, str) else item for item in value]
    return _parse(value, warnings)
