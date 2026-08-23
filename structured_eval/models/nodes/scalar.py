"""`ScalarNode` — a leaf position in the evaluation tree."""

from __future__ import annotations

from structured_eval.models.nodes.base import EvalNode


class ScalarNode(EvalNode):
    """A leaf node: a single comparable value.

    There is no pre-computed `similarity` — comparison *is* a metric. The match
    criterion is the node's `key_metric`, defined on `EvalNode`.
    """
