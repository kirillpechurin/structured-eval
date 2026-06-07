from __future__ import annotations

from typing import Any

from structured_eval.core.config import EvalConfig


def compute_faithfulness(
    actual: dict[str, Any],
    source: str,
    config: EvalConfig,
) -> tuple[float, list[str]]:
    """Compute faithfulness score using L1 substring check.

    For each leaf field in actual, checks whether the string representation
    of its value appears as a case-insensitive substring of source.

    Fields whose top-level key has FieldConfig(derived=True) are excluded
    from the check — derived values are computed from other fields and are
    not expected to appear verbatim in the source.

    Returns:
        Tuple of (score, hallucinated_fields).
        score: fraction of checked fields found in source. 1.0 when no
               fields are checkable (all derived or actual is empty).
        hallucinated_fields: paths of fields whose value was not found.

    Known limitation: only top-level FieldConfig.derived flags are respected.
    Nested derived fields (e.g. invoice.total inside an object) require v0.2.
    """
    source_lower = source.lower()
    field_configs = config.fields or {}
    scores: dict[str, float] = {}

    _collect_scores(actual, source_lower, field_configs, prefix="", out=scores)

    if not scores:
        return 1.0, []

    hallucinated = [k for k, s in scores.items() if s == 0.0]
    return sum(scores.values()) / len(scores), hallucinated


def _collect_scores(
    obj: dict[str, Any],
    source_lower: str,
    field_configs: dict[str, Any],
    prefix: str,
    out: dict[str, float],
) -> None:
    for key, value in obj.items():
        path = f"{prefix}.{key}" if prefix else key
        # Derived check: only top-level keys are looked up in field_configs.
        top_key = path.split(".")[0].split("[")[0]
        fc = field_configs.get(top_key)
        if fc and fc.derived:
            continue

        if isinstance(value, dict):
            _collect_scores(value, source_lower, field_configs, path, out)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                item_path = f"{path}[{i}]"
                if isinstance(item, dict):
                    _collect_scores(item, source_lower, field_configs, item_path, out)
                elif item is not None:
                    out[item_path] = 1.0 if str(item).lower() in source_lower else 0.0
        elif value is not None:
            out[path] = 1.0 if str(value).lower() in source_lower else 0.0
