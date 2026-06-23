# evaluate_consistency

`evaluate_consistency` measures **run-to-run stability** — how much a model's output
wobbles when you run the *same* prompt several times. It returns a
`ConsistencyReport` telling you which fields stay put and which drift.

The signature — the repeated runs, an optional config, and the variance bar that
splits stable from unstable:

```python
from structured_eval import evaluate_consistency

evaluate_consistency(runs, config=None, *, variance_threshold=0.05)
```

`runs` are several outputs for one input, each a `Sample`. They usually share an
`expected` (so each run is scored against the same reference); a field's *variance*
across those scores is what consistency is about.

## Checking stability

The examples below run against three outputs of one course-extraction prompt:
`course_id` and `title` come back identical every time, but `duration_hours` drifts
(12 → 10 → 14):

```python
from structured_eval import evaluate_consistency, Sample, EvalConfig, FieldConfig, NumericCloseness

expected = {"course_id": "COURSE-101", "title": "Introduction to Python", "duration_hours": 12}
config = EvalConfig(fields={"duration_hours": FieldConfig(metrics=[NumericCloseness()])})

runs = [
    Sample(actual={**expected, "duration_hours": 12}, expected=expected),
    Sample(actual={**expected, "duration_hours": 10}, expected=expected),
    Sample(actual={**expected, "duration_hours": 14}, expected=expected),
]

report = evaluate_consistency(runs, config, variance_threshold=0.001)
```

## Stable vs unstable fields

A field is **stable** when its score varies at most `variance_threshold` across
runs, otherwise **unstable**. `field_variance` gives the raw number per path:

```python
report.stable_fields     # ['course_id', 'title']  — identical every run
report.unstable_fields   # ['duration_hours']      — drifts past the bar

report.field_variance    # {'course_id': 0.0, 'title': 0.0, 'duration_hours': 0.00542}
```

The bar is yours to set: with the default `variance_threshold=0.05`,
`duration_hours` (variance `0.0054`) would count as stable. Lower the bar to demand
tighter agreement.

## Aggregate stability

Two top-level numbers summarise the whole document: the mean key-metric score and
how much that score itself varied between runs:

```python
report.mean_score        # 0.966 — averaged over the runs
report.score_variance    # 0.0006 — variance of the document score across runs
```

## Per-run reports

`per_run` is a list of `EvalReport`s — one per run, in order — each exactly what
[`evaluate`](evaluate.md#the-report) returns, so you can drill into any single run:

```python
for i, r in enumerate(report.per_run):
    print(f"run {i}  score {round(r.score, 3)}  duration={r.field_scores['duration_hours'].actual}")

# the three runs that produced the variance above:
# run 0  score 1.0    duration=12
# run 1  score 0.944  duration=10
# run 2  score 0.952  duration=14
```

`print_summary()` renders the split and the per-field variance:

```python
report.print_summary()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#   CONSISTENCY   3 runs
#   mean score       0.97
#   score variance   0.0006
#   stable           course_id, title
#   unstable         duration_hours
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#   Field           variance
#   ──────────────  ────────
#   duration_hours    0.0054
#   course_id         0.0000
#   title             0.0000
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Next

- [`evaluate`](evaluate.md) — the single-document report each `per_run` entry is.
- [`evaluate_batch`](evaluate-batch.md) — score a dataset of different documents.
- [The evaluation model](../core-concepts/evaluation-model.md) — the shared pipeline.
