from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from structured_eval.nodes.base import EvalNode


@dataclass
class FieldPair:
    """A matched (actual, expected) leaf pair and its matcher verdict.

    ``similarity`` is the raw output of ``matcher.similarity()`` (0.0–1.0).
    It is *not* a final score — each metric interprets it on its own terms.
    """

    actual: Any
    expected: Any
    matcher: Any  # Matcher — forward ref, matchers/ lands in Stage 4
    similarity: float


@dataclass
class ScalarNode(EvalNode):
    """A leaf node: a single comparable value."""

    pair: FieldPair
    metric_results: dict[str, float] = field(default_factory=dict)
