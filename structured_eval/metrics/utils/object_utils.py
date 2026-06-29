"""Verdicts for object metrics: matched fields → ``(score, threshold, weight)``.

A parent object does not re-compare its children; it reads each matched child's
already-computed representative score (``node.representative``) and pairs it with
the bar it must clear and the weight it carries. Those triples feed
``calculate.prf_counts``.

``score_policy`` (on ``ObjectF1`` / ``ObjectAccuracy`` / …) overrides the
criterion for a named field — a metric instance or its registered name, run on
that child via ``MetricInvoker`` (so it works for any child kind, not only
scalars). ``thresholds`` may be a per-field dict or a single float.

``weight_mode`` (see ``calculate.WeightMode``) decides each child's weight:
``NONE`` → ``1.0`` (plain counts), ``PROPORTIONAL`` → the child's configured
``weight``. Missing (FN) and spurious (FP) children are weighted the same way
via ``missing_weight`` / ``spurious_weight``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from structured_eval.metrics.base import resolve_metric
from structured_eval.metrics.invoker import MetricInvoker
from structured_eval.metrics.utils.calculate import WeightMode

if TYPE_CHECKING:
    from structured_eval.model.nodes.base import EvalNode
    from structured_eval.model.nodes.object_node import ObjectNode


def leaf_name(path: str) -> str:
    """Last path segment without any trailing index, e.g. ``"a.b[0]"`` → ``"b"``."""
    return path.rsplit(".", 1)[-1].split("[", 1)[0]


def _resolve_threshold(thresholds: float | dict[str, float] | None, name: str, fallback: float) -> float:
    if isinstance(thresholds, dict):
        return float(thresholds.get(name, fallback))
    if thresholds is not None:
        return float(thresholds)
    return fallback


def _weight_of(child: EvalNode, weight_mode: WeightMode) -> float:
    return child.weight if weight_mode == WeightMode.PROPORTIONAL else 1.0


def matched_verdicts(
    node: ObjectNode,
    score_policy: dict[str, Any] | None = None,
    thresholds: float | dict[str, float] | None = None,
    weight_mode: WeightMode = WeightMode.PROPORTIONAL,
) -> list[tuple[float, float, float]]:
    """``(score, threshold, weight)`` for each matched child of an object.

    Each child contributes its representative score (any node type — scalars and
    nested objects/arrays alike). ``score_policy`` overrides the criterion for a
    named child, re-scoring it with the policy metric (any node kind).
    """
    out: list[tuple[float, float, float]] = []
    for child in node.matched:
        name = leaf_name(child.path)
        spec = (score_policy or {}).get(name)
        if spec is not None:
            score = MetricInvoker(resolve_metric(spec)).scalar_on_node(child)
        else:
            score = child.representative
        threshold = _resolve_threshold(thresholds, name, child.threshold)
        out.append((score, threshold, _weight_of(child, weight_mode)))
    return out


def missing_weight(node: ObjectNode, weight_mode: WeightMode = WeightMode.PROPORTIONAL) -> float:
    """Summed weight of the object's missing (FN) children (count when uniform)."""
    return sum(_weight_of(node.children[name], weight_mode) for name in node.missing)


def spurious_weight(node: ObjectNode, weight_mode: WeightMode = WeightMode.PROPORTIONAL) -> float:
    """Summed weight of the object's spurious (FP) children (count when uniform)."""
    return sum(_weight_of(node.children[name], weight_mode) for name in node.spurious)
