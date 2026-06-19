"""Representative scores and verdicts shared by the aggregating metrics.

Every node designates one metric as its ``key_metric`` — its *representative*
score. A parent object/array does not re-compare its children; it reads each
matched child's already-computed representative score (``repr_score``) and
aggregates those. This is what makes the framework fully recursive: an object
nested in an object contributes exactly the same way a scalar does.

``structural_score`` is the zero-config fallback used when a node carries no
metrics of its own — the recursive soft mean over its matched children (a
missing/missed child counts 0).

``score_policy`` (on ``ObjectF1``/``ObjectAccuracy``/…) overrides the criterion
for a named *scalar* field: a ``FieldMetric`` instance or its registered name.
``thresholds`` may be a per-field dict or a single float for all fields.
"""

from __future__ import annotations

from typing import Any

from structured_eval.metrics.base import FieldMetric, Metric, resolve_metric
from structured_eval.metrics.exact import ExactMatch
from structured_eval.model.nodes.array_node import ArrayNode
from structured_eval.model.nodes.base import EvalNode
from structured_eval.model.nodes.object_node import ObjectNode
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


def repr_score(node: EvalNode) -> float:
    """The node's representative score: its ``key_metric`` result if computed.

    Falls back to ``structural_score`` when the key metric hasn't run yet or no
    key metric is set (so callers outside the engine still get a value).
    """
    # TODO: That's ok...
    km = node.key_metric
    if km is not None:
        value = node.metric_results.get(km.name)
        if value is not None:
            return float(value)
    return structural_score(node)


def structural_score(node: EvalNode) -> float:
    """Recursive soft mean of a node's correctness, ignoring its key metric.

    Scalar → its match-criterion verdict; object → mean of matched children's
    representative scores over (matched + missing); array → mean over
    (items + missed). An empty/fully-missing node is vacuously ``1.0``.
    """
    # TODO: Why should we still have this?
    if isinstance(node, ScalarNode):
        metric = node.key_metric if isinstance(node.key_metric, FieldMetric) else ExactMatch()
        raw = metric.score(node.actual, node.expected)
        assert not isinstance(raw, dict)  # a scalar criterion returns a scalar
        return float(raw)
    if isinstance(node, ObjectNode):
        denom = len(node.matched) + len(node.missing)
        if denom == 0:
            return 1.0
        return sum(repr_score(child) for child in node.matched) / denom
    if isinstance(node, ArrayNode):
        n_missing = len(node.match_result.missed) if node.match_result else 0
        denom = len(node.items) + n_missing
        if denom == 0:
            return 1.0
        return sum(repr_score(item) for item in node.items) / denom
    return 0.0


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
            score = repr_score(child)
        out.append((score, _resolve_threshold(thresholds, name, child.threshold)))
    return out
