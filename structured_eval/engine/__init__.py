"""The evaluation pipeline: parse, build the tree, compute metrics, report.

`Evaluator` owns the sequence; each phase is its own class so it can be tested
and swapped in isolation. `BatchAggregator` rolls per-document reports into one.
"""

from structured_eval.engine.aggregator import BatchAggregator
from structured_eval.engine.evaluator import Evaluator
from structured_eval.engine.metric_runner import MetricRunner
from structured_eval.engine.parser import Parser
from structured_eval.engine.report_builder import ReportBuilder
from structured_eval.engine.tree_builder import TreeBuilder

__all__ = [
    "BatchAggregator",
    "Evaluator",
    "MetricRunner",
    "Parser",
    "ReportBuilder",
    "TreeBuilder",
]
