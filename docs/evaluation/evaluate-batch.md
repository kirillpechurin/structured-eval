# evaluate_batch

`evaluate_batch` scores a **dataset** — many documents at once — and returns a
`BatchEvalReport`: every document's own report plus batch-level aggregates. Use it
when you care about how a model does *across* a set, not just on one output.

The signature — a list of `Sample`s and an optional shared config:

```python
from structured_eval import evaluate_batch

evaluate_batch(samples, config=None)
```

Each `Sample` carries its own `actual` / `expected` / `source` / `id`, and the same
`config` applies to all of them.

## Scoring a dataset

The examples below all run against this small batch of course extractions — two
clean documents and one that fails to parse:

```python
from structured_eval import evaluate_batch
from structured_eval.models import EvalConfig, FieldConfig, Sample
from structured_eval.metrics import NumericCloseness, TokenF1

config = EvalConfig(fields={
    "title": FieldConfig(metrics=[TokenF1()]),
    "duration_hours": FieldConfig(metrics=[NumericCloseness()]),
})

samples = [
    Sample(id="c1",
           actual={"title": "Intro to Python", "duration_hours": 10},
           expected={"title": "Introduction to Python", "duration_hours": 12}),
    Sample(id="c2",
           actual={"title": "Data Structures", "duration_hours": 20},
           expected={"title": "Data Structures", "duration_hours": 20}),
    Sample(id="c3",
           actual="{bad json",
           expected={"title": "X", "duration_hours": 5}),
]

report = evaluate_batch(samples, config)
```

## Batch aggregates

The top-level fields summarise the whole set. `score` is the mean key-metric score,
`metrics` averages each document-level metric, and the rates tell you how many
documents were perfect or unparseable (computed over parsed samples):

```python
report.score                    # 0.875 — mean key-metric across parsed samples
report.score_label              # 'mean_score'
report.metrics                  # {'object_accuracy': 0.875, 'token_f1': 0.833, ...}
report.perfect_response_rate    # 0.333 — only c2 passed every field
report.parse_error_rate         # 0.333 — c3 failed to parse
```

## Per-sample reports

`per_sample` is a list of `EvalReport`s — one per input `Sample`, in the same
order. Each is exactly the report `evaluate` returns, so everything on the
[`evaluate` report page](evaluate.md#the-report) applies (field scores, metrics,
`failed_fields()`, …):

```python
for sample, r in zip(samples, report.per_sample):
    if r.parse_error:
        print(f"{sample.id:<4} parse error")
    else:
        print(f"{sample.id:<4} {round(r.score, 3)}")

# one row per document, aligned with the input order:
# c1   0.75
# c2   1.0
# c3   parse error
```

A parse failure doesn't abort the batch: that sample's report has
`parse_error=True` and a `None` score, and it's simply excluded from the
aggregates.

## Field breakdown

`field_breakdown()` pivots the batch the other way — **per field across all
documents** — so you can spot which fields are consistently weak. Each path reports
mean / min / max / p95 and a `fail_rate` (fraction of samples scoring below the
field's threshold):

```python
for path, stats in report.field_breakdown().items():
    print(path, {k: round(v, 3) for k, v in stats.items()})

# per-path statistics over the parsed samples:
# $               {'mean': 0.875, 'min': 0.75,  'max': 1.0, 'p95': 0.988, 'fail_rate': 0.5}
# title           {'mean': 0.833, 'min': 0.667, 'max': 1.0, 'p95': 0.983, 'fail_rate': 0.5}
# duration_hours  {'mean': 0.917, 'min': 0.833, 'max': 1.0, 'p95': 0.992, 'fail_rate': 0.5}
```

Pass an explicit `threshold` to score every field against the same bar:
`report.field_breakdown(threshold=0.9)`.

A `print_summary()` renders the aggregates and the field breakdown as a table:

```python
report.print_summary()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#   BATCH   3 samples
#   mean mean_score   0.88
#   perfect_response_rate   0.33
#   parse_error_rate        0.33
#
#   object_accuracy   0.88     numeric_closeness 0.92
#   token_f1          0.83
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#   Field breakdown (worst first)
#   Field           mean   p95  fail_rate
#   ──────────────  ────  ────  ─────────
#   $               0.88  0.99       0.50
#   duration_hours  0.92  0.99       0.50
#   title           0.83  0.98       0.50
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Next

- [`evaluate`](evaluate.md) — the single-document report each `per_sample` entry is.
- [`evaluate_consistency`](evaluate-consistency.md) — repeated runs of one prompt.
- [The evaluation model](../core-concepts/evaluation-model.md) — the shared pipeline.
