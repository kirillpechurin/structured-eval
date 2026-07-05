# The evaluation functions

structured-eval has three entry points. They share one engine and the same
[evaluation model](../core-concepts/evaluation-model.md) — they differ only in
*what* you evaluate and *which* report you get back.

| Function                                          | Use it when                                                                   | Input                        | Returns             |
|---------------------------------------------------|-------------------------------------------------------------------------------|------------------------------|---------------------|
| [`evaluate`](evaluate.md)                         | scoring one output against a reference (or providing source for some metrics) | one document or `Sample`     | `EvalReport`        |
| [`evaluate_batch`](evaluate-batch.md)             | scoring a whole dataset and reading aggregates                                | `list[Sample]`               | `BatchEvalReport`   |
| [`evaluate_consistency`](evaluate-consistency.md) | checking how stable one prompt is across repeated runs                        | `list[Sample]` (the repeats) | `ConsistencyReport` |

## One document — `evaluate`

```python
from structured_eval import evaluate

report = evaluate(actual, expected, config, source=None)  # → EvalReport
report.score  # a float in [0, 1]
```

A bare `list` is treated as one document with an array root, **not** a batch.
See [evaluate](evaluate.md) for input forms, providing `source` for some
metrics, and the report.

## A dataset — `evaluate_batch`

```python
from structured_eval import evaluate_batch
from structured_eval.models import Sample

samples = [Sample(actual=a, expected=e) for a, e in pairs]
report = evaluate_batch(samples, config)  # → BatchEvalReport
report.per_sample  # per-sample reports + aggregates
```

See [evaluate_batch](evaluate-batch.md) for `per_sample`, batch metrics, and
`field_breakdown()`.

## Repeated runs — `evaluate_consistency`

```python
from structured_eval import evaluate_consistency
from structured_eval.models import Sample

runs = [Sample(actual=output) for output in repeated_outputs]
report = evaluate_consistency(runs, config, variance_threshold=0.05)  # → ConsistencyReport
report.stable_fields, report.unstable_fields  # per-field stability
```

See [evaluate_consistency](evaluate-consistency.md) for variance, stable /
unstable fields, and score variance.

## Next

- [`evaluate`](evaluate.md) — one document, input forms, providing `source`.
- [`evaluate_batch`](evaluate-batch.md) — datasets and aggregates.
- [`evaluate_consistency`](evaluate-consistency.md) — run-to-run stability.
- [The evaluation model](../core-concepts/evaluation-model.md) — the shared
  pipeline behind all three.
