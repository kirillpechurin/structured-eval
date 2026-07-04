from __future__ import annotations

from structured_eval.models.nodes.base import EvalNode


class ScalarNode(EvalNode):
    """A leaf node: a single comparable value.

    In v3 there is no pre-computed ``similarity`` — comparison *is* a metric.
    The match criterion is the node's ``key_metric`` (defined on ``EvalNode``):
    its representative score, defaulting to ``MeanScore`` over the node's field
    metrics (a lone ``ExactMatch`` when none are configured). ``threshold`` is
    the bar that score must clear to count as a true positive.
    """
