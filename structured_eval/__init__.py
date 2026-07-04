"""structured_eval — field-level evaluation of structured LLM outputs.

The top level exposes only the entrypoints. Everything else lives one level
down, imported explicitly from its subsystem:

- ``structured_eval.models``   — user-facing data models: ``Sample``,
  ``EvalConfig`` (+ the ``*FieldConfig`` family & policies), ``EvalReport`` /
  ``BatchEvalReport`` / ``ConsistencyReport``. Lower-level model pieces live in
  precise submodules (``models.nodes`` / ``models.result`` /
  ``models.metric_result`` / ``models.context``).
- ``structured_eval.metrics``  — every metric plus the base hierarchy
  (``Metric`` / ``FieldMetric`` / …), ``resolve_metric``, and the rule DSL
  (``Rule`` / ``RulePassRate``).
- ``structured_eval.alignment`` / ``.formats`` / ``.utils`` — supporting
  machinery (array alignment, parsers, ``flatten`` / ``structured_diff``).

``evaluate`` / ``evaluate_batch`` / ``evaluate_consistency`` are thin wrappers
over ``engine.Evaluator``.
"""

from structured_eval.api import evaluate, evaluate_batch, evaluate_consistency

__all__ = [
    "evaluate",
    "evaluate_batch",
    "evaluate_consistency",
]
