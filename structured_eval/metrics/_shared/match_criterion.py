"""Resolve the match criterion for a scalar field under an object/array metric.

Resolution order (technical_details_v3 §3) for field ``X``:

1. ``score_policy[X]`` (+ ``thresholds[X]``) declared on the aggregating metric;
2. else the field's own ``key_metric`` (+ its ``threshold``);
3. else the default ``ExactMatch()`` @ ``threshold=1.0``.

``score_policy`` values may be a ``FieldMetric`` instance or its registered name
string. ``thresholds`` may be a per-field dict or a single float for all fields.
"""

from __future__ import annotations

from typing import Any

from structured_eval.metrics.base import FieldMetric, get_metric_class
from structured_eval.metrics.exact import ExactMatch
from structured_eval.model.nodes.scalar import ScalarNode


def leaf_name(path: str) -> str:
    """Last path segment without any trailing index, e.g. ``"a.b[0]"`` → ``"b"``."""
    return path.rsplit(".", 1)[-1].split("[", 1)[0]


def _resolve_metric(spec: Any) -> FieldMetric:
    if isinstance(spec, str):
        instance = get_metric_class(spec)()
        assert isinstance(instance, FieldMetric)
        return instance
    assert isinstance(spec, FieldMetric)
    return spec


def _resolve_threshold(thresholds: Any, name: str, fallback: float) -> float:
    if isinstance(thresholds, dict):
        return float(thresholds.get(name, fallback))
    if thresholds is not None:
        return float(thresholds)
    return fallback


def field_verdict(
    node: ScalarNode,
    score_policy: dict[str, Any] | None = None,
    thresholds: Any = None,
) -> tuple[float, float]:
    """Return ``(score, threshold)`` for one matched scalar field."""
    name = leaf_name(node.path)

    spec = (score_policy or {}).get(name)
    if spec is not None:
        metric = _resolve_metric(spec)
    elif node.key_metric is not None:
        metric = node.key_metric
    else:
        metric = ExactMatch()

    raw = metric.score(node.actual, node.expected)
    assert not isinstance(raw, dict)  # a match criterion is a scalar comparison
    score = float(raw)
    threshold = _resolve_threshold(thresholds, name, getattr(node, "threshold", 1.0))
    return score, threshold


def matched_scalar_verdicts(
    node: Any,
    score_policy: dict[str, Any] | None = None,
    thresholds: Any = None,
) -> list[tuple[float, float]]:
    """``(score, threshold)`` for each matched scalar child of an object/array."""
    return [
        field_verdict(child, score_policy, thresholds)
        for child in node.matched
        if isinstance(child, ScalarNode)
    ]
