"""Verdicts for object metrics: matched fields → ``(score, threshold)`` pairs.

A parent object does not re-compare its children; it reads each matched child's
already-computed representative score (``node.representative``) and pairs it with
the bar it must clear. Those pairs feed ``calculate.prf_counts``.

``score_policy`` (on ``ObjectF1`` / ``ObjectAccuracy`` / …) overrides the
criterion for a named field — a metric instance or its registered name, run on
that child via ``MetricInvoker`` (so it works for any child kind, not only
scalars). ``thresholds`` may be a per-field dict or a single float.
"""

from __future__ import annotations

from typing import Any

from structured_eval.metrics.base import resolve_metric
from structured_eval.metrics.invoker import MetricInvoker
from structured_eval.model.nodes.object_node import ObjectNode


def leaf_name(path: str) -> str:
    """Last path segment without any trailing index, e.g. ``"a.b[0]"`` → ``"b"``."""
    return path.rsplit(".", 1)[-1].split("[", 1)[0]


def _resolve_threshold(thresholds: Any, name: str, fallback: float) -> float:
    if isinstance(thresholds, dict):
        return float(thresholds.get(name, fallback))
    if thresholds is not None:
        return float(thresholds)
    return fallback


def matched_verdicts(
    node: ObjectNode,
    score_policy: dict[str, Any] | None = None,
    thresholds: Any = None,
) -> list[tuple[float, float]]:
    """``(score, threshold)`` for each matched child of an object.

    Each child contributes its representative score (any node type — scalars and
    nested objects/arrays alike). ``score_policy`` overrides the criterion for a
    named child, re-scoring it with the policy metric (any node kind).
    """
    out: list[tuple[float, float]] = []
    for child in node.matched:
        name = leaf_name(child.path)
        spec = (score_policy or {}).get(name)
        if spec is not None:
            score = MetricInvoker(resolve_metric(spec)).scalar_on_node(child)
        else:
            score = child.representative
        out.append((score, _resolve_threshold(thresholds, name, child.threshold)))
    return out
