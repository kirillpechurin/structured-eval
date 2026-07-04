# CoverageLeafScore

|            |                       |
|------------|-----------------------|
| **Class**  | `CoverageLeafScore`   |
| **Key**    | `coverage_leaf_score` |
| **Branch** | root (whole document) |
| **Needs**  | `expected`            |

## What it measures

**Completeness** — of all the scalar fields that *should* carry a value, what fraction did
the model actually fill in. It only asks "is the field present?", not whether the value is
right. This is the presence half that complements value accuracy: you can't get a field
right if you never produced it.

Like [`OverallLeafScore`](overall-leaf-score.md) it flattens to scalar leaves across the
whole tree, but it counts presence (0/1 per leaf), not correctness.

## Parameters

None — `CoverageLeafScore()`. Leaves are counted equally (no weighting).

## How it's computed

```text
for every scalar leaf in the tree:
    if it is expected to have a value (expected is not null):
        expected_count += 1
        if actual has a value (actual is not null):
            covered_count += 1

score = covered_count / expected_count        # 1.0 if nothing is expected
```

## Example

Three expected leaves (`title`, `meta.level`, `meta.hours`); the model omitted `meta.hours`:

```python
from structured_eval import evaluate
from structured_eval.models import EvalConfig
from structured_eval.metrics.coverage_leaf_score import CoverageLeafScore

config = EvalConfig(key_metric=CoverageLeafScore())
report = evaluate(
    {"title": "ML", "meta": {"level": "advanced"}},               # hours missing
    {"title": "ML", "meta": {"level": "advanced", "hours": 40}},
    config,
)

# present: title, meta.level → 2 of 3 expected
report.score                                              # 0.6667
report.score_label                                        # "coverage_leaf_score"
```

It is **value-blind** — a document that fills every field but gets every value wrong still
scores `1.0`:

```python
report = evaluate(
    {"title": "WRONG", "meta": {"level": "x", "hours": 1}},
    {"title": "ML", "meta": {"level": "advanced", "hours": 40}},
    config,
)
report.score                                              # 1.0 — all present, none correct
```

## Edge cases

- **Value-blind** — only presence counts; pair with [`OverallLeafScore`](overall-leaf-score.md)
  or the object/value metrics to judge correctness.
- **Unweighted** — every leaf counts 1 (unlike `OverallLeafScore`, which honors
  `FieldConfig.weight`).
- **Array `missed` elements aren't counted** — an expected element that was never produced
  has no leaf node; array completeness belongs to the array metrics.
- **Nothing expected** → vacuously `1.0`.

## See also

- [`OverallLeafScore`](overall-leaf-score.md) — correctness over the same leaves (the other half).
- [`SchemaValidity`](schema-validity.md) — structural validity, including required fields.
- [`Presence`](presence.md) — the per-field version of "is this filled in?".
- [The metric catalog](../index.md) — all metrics and the return-shape model.
