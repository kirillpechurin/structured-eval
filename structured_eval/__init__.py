"""structured_eval — field-level evaluation of structured LLM outputs.

The top level exposes only `evaluate`, `evaluate_batch` and
`evaluate_consistency`. Everything else is imported from its own subsystem:

- `structured_eval.models` — `Sample`, `EvalConfig`, `EvalReport` and the rest
  of the data layer.
- `structured_eval.metrics` — every metric, the base hierarchy, `resolve_metric`
  and the rule DSL.
- `structured_eval.alignment` / `.formats` / `.utils` — array alignment,
  parsers, `flatten` / `structured_diff`.
"""

from structured_eval.api import evaluate, evaluate_batch, evaluate_consistency

__all__ = [
    "evaluate",
    "evaluate_batch",
    "evaluate_consistency",
]
