"""Metric values and the shapes they travel in.

Split by what each piece is about: ``result`` — one metric's value at one node;
``collection`` — one named metric's values across the tree, as the report sees
them; ``judge`` — what an LLM judge reports, independent of *what* it judges.
"""

from structured_eval.models.metrics.collection import MetricCollection
from structured_eval.models.metrics.judge import FieldJudgeVerdict, JudgeVerdict
from structured_eval.models.metrics.result import MetricResult

__all__ = [
    "FieldJudgeVerdict",
    "JudgeVerdict",
    "MetricCollection",
    "MetricResult",
]
