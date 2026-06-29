# StructuralSimilarity

|            |                         |
|------------|-------------------------|
| **Class**  | `StructuralSimilarity`  |
| **Key**    | `structural_similarity` |
| **Branch** | root (`$` only)         |
| **Needs**  | `expected`              |

## What it measures

How much two documents share the **same shape**, ignoring their values — the Jaccard
overlap of their structural paths. It answers "did the model produce the right skeleton"
(the right keys, nesting, and list positions) independently of whether the values inside
are correct. A useful top-line signal when the *structure* is what's drifting — extra
keys, dropped sections, wrong nesting — before you dig into value correctness with the
field/object/array metrics.

## Parameters

None — `StructuralSimilarity()`. As a root metric it runs once on the whole document; add
it via `EvalConfig(metrics=[...])`.

## How it's computed

```text
score = |paths(actual) ∩ paths(expected)| / |paths(actual) ∪ paths(expected)|
```

`paths(...)` enumerates every structural path — each dict key, each list index, and every
nested sub-path (containers **and** leaves), e.g. `{"a": {"b": 1}}` → `{"a", "a.b"}`.
Values are ignored; only which paths exist matters.

## Example

```python
from structured_eval import evaluate, EvalConfig, StructuralSimilarity

config = EvalConfig(metrics=[StructuralSimilarity()])

# same keys, different values → identical shape
report = evaluate(
    {"title": "Python 101", "level": "beginner"},
    {"title": "Algorithms", "level": "advanced"},
    config,
)
report.field_scores["$"].metrics["structural_similarity"]   # 1.0

# a key dropped, a key added → shapes only partly overlap
report = evaluate(
    {"title": "Python 101", "level": "beginner"},
    {"title": "Python 101", "credits": 3},
    config,
)
# paths {title, level} vs {title, credits}: ∩=1, ∪=3 → 0.333
report.field_scores["$"].metrics["structural_similarity"]   # 0.333
```

## Edge cases

- **Value-blind** — two documents with the same keys score `1.0` no matter how wrong the
  values are. Combine it with value metrics; it is not a quality score on its own.
- **Both empty → `1.0`**, **exactly one side empty → `0.0`**.
- **Containers count** — intermediate paths (`a`, `a.items`) are included, so two
  documents that agree on nesting but differ only in leaf names still partly overlap.
- **List indices are positional** — a path is `tags[0]`, `tags[1]`, …; reordering a list
  keeps the same index paths, but a length change adds/removes them.

## See also

- [`CoverageLeafScore`](coverage-leaf-score.md) — fraction of *expected* leaves present
  (completeness, also value-blind).
- [`SchemaValidity`](schema-validity.md) — shape conformance against an explicit schema.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
