from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from structured_eval.nodes.base import EvalNode


@dataclass
class ScalarNode(EvalNode):
    """A leaf node: a single comparable value.

    In v3 there is no pre-computed ``similarity`` — comparison *is* a metric
    (see ``metrics/field/``). ``key_metric`` is the field metric designated as
    the match criterion for the parent object/array (``None`` → ``ExactMatch``
    is used at resolution time); ``threshold`` is the bar it must clear to count
    as a true positive. ``metric_results`` holds each requested metric's value
    for this field.
    """

    key_metric: Any = None  # FieldMetric used as the parent's match criterion
    threshold: float = 1.0
    metric_results: dict[str, float] = field(default_factory=dict)
