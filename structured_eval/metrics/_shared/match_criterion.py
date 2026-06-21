"""Field verdicts shared by the aggregating object/array metrics.

Every node designates one metric as its ``key_metric`` — its *representative*
score, exposed as ``node.representative``. A parent object/array does not
re-compare its children; it reads each matched child's already-computed
representative and aggregates those. This is what makes the framework fully
recursive: an object nested in an object contributes exactly the way a scalar
does.

``score_policy`` (on ``ObjectF1``/``ObjectAccuracy``/…) overrides the criterion
for a named *scalar* field: a ``FieldMetric`` instance or its registered name.
``thresholds`` may be a per-field dict or a single float for all fields.
"""

from __future__ import annotations

from typing import Any

from structured_eval.metrics.base import Metric, resolve_metric
from structured_eval.model.nodes.scalar import ScalarNode


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
    node: Any,
    score_policy: dict[str, Any] | None = None,
    thresholds: Any = None,
) -> list[tuple[float, float]]:
    """``(score, threshold)`` for each matched child of an object/array.

    Each child contributes its representative score (any node type — scalars and
    nested objects/arrays alike). ``score_policy`` overrides the criterion for a
    named scalar child, re-scoring it with the policy metric.
    """
    # TODO: It can be a class method of general metric ObjectMatchedMetric (use abstraction)
    out: list[tuple[float, float]] = []
    for child in node.matched:
        name = leaf_name(child.path)
        spec = (score_policy or {}).get(name)
        if spec is not None and isinstance(child, ScalarNode):
            metric = resolve_metric(spec)
            assert isinstance(metric, Metric)  # a match criterion compares via score()
            raw = metric.score(child.actual, child.expected)
            assert not isinstance(raw, dict)  # a match criterion is a scalar comparison
            score = float(raw)
        else:
            score = child.representative
        out.append((score, _resolve_threshold(thresholds, name, child.threshold)))
    return out
