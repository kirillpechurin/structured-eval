# Comparison is a metric

[The evaluation model](evaluation-model.md) introduced the idea in one sentence:
a metric takes a node's `(actual, expected)` and returns a score. This page takes
that design choice seriously and follows it to its consequences — why there is no
separate "matcher", why a field can carry several scorers at once, why aggregation
and even array alignment are *also* just metrics, and how a node turns its many
metrics into the one number it reports.

## The thesis

Many evaluation tools split scoring into two stages: a **matcher** first computes
a similarity between actual and expected, then a separate **scorer** turns those
similarities into numbers. `structured-eval` collapses that into a single idea:

> **A comparison *is* a metric.** There is no separate matcher and no
> pre-computed `similarity` attribute on a node. A metric takes a node's
> `(actual, expected)` and returns a score in `[0, 1]` — that is the whole
> contract.

Why it matters: a `similarity` baked into the node forces *one* comparison policy
per field, decided before you know what you want to ask. Making comparison a
metric lets a field carry several policies, computed lazily, and lets the same
contract describe scalars, objects, arrays, and the whole document. Everything
below is a consequence of that.

## Consequence 1 — a node carries *several* comparisons

Because a comparison is just a metric, and a node owns a **list** of them, one
field can be scored by several metrics at once — a cheap exact check *and* an
expensive fuzzy one — each scoring the very same `(actual, expected)` pair, none
computed speculatively.

```python
from structured_eval import evaluate
from structured_eval.models import EvalConfig, FieldConfig
from structured_eval.metrics import ExactMatch, Fuzzy, TokenF1

report = evaluate(
    {"title": "Intro to Python"},
    {"title": "Introduction to Python"},
    EvalConfig(fields={"title": FieldConfig(metrics=[ExactMatch(), TokenF1(), Fuzzy()])}),
)

fs = report.field_scores["title"]
float(fs.metrics["exact_match"])   # 0.0    — the strings differ
float(fs.metrics["token_f1"])      # 0.6667 — they share "to python"
float(fs.metrics["fuzzy"])         # 0.8108 — edit-distance similarity
```

In a matcher-based design you would commit to one similarity up front. Here you
attach as many lenses as you want and decide *later* which one speaks for the
field (see [the representative score](#the-representative-score-key_metric)).

### Two configurations of the *same* metric

Results are keyed by the metric's name, so two `Numeric()` instances on one field
would both key on `"numeric"` and the second would overwrite the first. Pass
`name=` to give an instance its own key:

```python
from structured_eval.metrics import Numeric

config = EvalConfig(fields={"total": FieldConfig(metrics=[
    Numeric(tolerance=0.001, name="strict"),
    Numeric(tolerance=0.1, name="loose"),
])})

report = evaluate({"total": 100.5}, {"total": 100.0}, config)
float(report.metrics["strict"].by_path["total"])  # 0.0 — 0.5% off, outside 0.1%
float(report.metrics["loose"].by_path["total"])   # 1.0 — inside 10%
```

Any metric accepts `name=`. It renames that *instance* only: the class keeps its
registered name, so `key_metric="numeric"` and other name-string lookups are
unaffected — though `key_metric="strict"` now works too. Metrics that return a
dict of sub-scores (`ObjectPRF1` → `precision` / `recall` / `f1`) write those
keys directly, so `name=` does not rename them.

## Consequence 2 — aggregation is also a metric

There is no privileged "aggregate" phase either. An `ObjectF1` or an
`ArrayAccuracy` is *also* just a metric — it merely reads its children's
already-computed scores instead of `(actual, expected)` directly. Same contract,
different input. This is why metrics run **post-order** (children before parents):
an aggregating parent reads each child's representative.

```python
from structured_eval import evaluate
from structured_eval.models import EvalConfig
from structured_eval.metrics import ObjectF1

report = evaluate(
    {"name": "Sarah", "experience_years": 5},
    {"name": "Sarah", "experience_years": 8},
    EvalConfig(metrics=[ObjectF1()]),
)

report.field_scores["name"].score              # 1.0  — child, computed first
report.field_scores["experience_years"].score  # 0.0
float(report.metrics["object_f1"].root())      # 0.5  — parent reads the two above
```

The same holds one level deeper, over a list of objects: each element is itself a
node with its own representative, and the array metric averages those.

```python
from structured_eval import evaluate
from structured_eval.models import ArrayFieldConfig, EvalConfig
from structured_eval.metrics import ArrayAccuracy

config = EvalConfig(fields={"instructors": ArrayFieldConfig(metrics=[ArrayAccuracy()])})
report = evaluate(
    {"instructors": [{"name": "Sarah", "years": 5}, {"name": "Tom", "years": 3}]},
    {"instructors": [{"name": "Sarah", "years": 5}, {"name": "Tom", "years": 9}]},
    config,
)

# each element is an ObjectNode with its own representative:
report.field_scores["instructors[0]"].score   # 1.0  — both fields match
report.field_scores["instructors[1]"].score   # 0.5  — name ok, years wrong
# the array metric averages those element representatives:
float(report.field_scores["instructors"].metrics["array_accuracy"])  # 0.75
```

`ArrayAccuracy` compares nothing by hand — it reads the already-computed
representatives of its element objects, which in turn read the representatives of
*their* fields. One contract, at every level of the tree.

## Consequence 3 — array alignment is a metric too

Even *pairing* array elements is a comparison. Before an array metric can score
elements, it has to decide which actual element answers which expected one — and
that decision is made by scoring `(actual_key, expected_key)` with an ordinary
metric.

```python
from structured_eval import evaluate
from structured_eval.models import ArrayFieldConfig, ArrayStrategy, EvalConfig

# elements arrive in a different order — pair them by the `id` key, not position
config = EvalConfig(fields={"items": ArrayFieldConfig(
    strategy=ArrayStrategy.BY_KEY,
    params={"key": "id"},
)})
report = evaluate(
    {"items": [{"id": "B", "qty": 2}, {"id": "A", "qty": 1}]},
    {"items": [{"id": "A", "qty": 1}, {"id": "B", "qty": 2}]},
    config,
)
report.array_matches["items"].matched   # [(0, 1), (1, 0)] — A↔A, B↔B despite the order
```

`ByKeyAligner` compares keys with a metric (`ExactMatch` by default), so element
pairing is the same comparison-is-a-metric idea applied to keys. The mechanics —
`by_index` / `by_key` / `hungarian`, and matched / missed / spurious — are in
[array alignment](array-alignment.md).

## The representative score (`key_metric`)

A node may carry many metrics but reports **one** number — its *representative*,
chosen by the node's `key_metric`. By default that is `MeanScore` (the mean of the
node's own metrics), but **any metric can be the representative**, so you can pick
one to speak for the node.

The representative is computed **last**, after the node's other metrics. That
ordering is what makes it powerful: a `key_metric` can read `node.metric_results`
and combine the already-computed scores however it likes — weight them, blend
them, mix them with its own logic.

```python
from structured_eval import evaluate
from structured_eval.models import EvalConfig, FieldConfig
from structured_eval.metrics import ExactMatch, Fuzzy, TokenF1
from structured_eval.metrics.base import AnyNodeMetric
from structured_eval.models.nodes.base import EvalNode

class WeightedBlend(AnyNodeMetric):
    """A representative that blends the field's other metrics by hand."""
    name = "weighted_blend"

    def compute(self, node: EvalNode) -> float:
        r = node.metric_results            # MetricResult is a float subclass
        return 0.2 * r["exact_match"] + 0.5 * r["fuzzy"] + 0.3 * r["token_f1"]

config = EvalConfig(fields={"title": FieldConfig(
    metrics=[ExactMatch(), Fuzzy(), TokenF1()],
    key_metric=WeightedBlend(),
)})
report = evaluate(
    {"title": "Intro to Python"},
    {"title": "Introduction to Python"},
    config,
)

fs = report.field_scores["title"]
float(fs.metrics["exact_match"]), float(fs.metrics["fuzzy"]), float(fs.metrics["token_f1"])
#  (0.0, 0.8108, 0.6667)
fs.score                          # 0.6054  — 0.2·0.0 + 0.5·0.8108 + 0.3·0.6667
fs.metrics["weighted_blend"]      # 0.6054  — also stored under its own name
```

`MeanScore` is simply the equal-weight special case of this same move. And because
`compute` receives the whole `node`, a representative need not lean on the other
metrics at all — it can score the field entirely its own way.

> Scope note: `report.score` / `report.score_label` are the **root** node's
> representative, not a field's. Above we only set a `key_metric` on `title`, so
> `fs.score` carries the blend while the root keeps its default. The root → report
> wiring is covered in
> [the evaluation model](evaluation-model.md#the-representative-score-key_metric).

## How a metric returns its score

Every metric obeys one return contract, normalized by the engine
(`MetricRunner._apply`) into entries in `node.metric_results`. A metric attaches
structured detail by **returning** it, never by mutating state.

| A metric returns      | Lands in the report as                                            |
|-----------------------|-------------------------------------------------------------------|
| `float`               | one entry under the metric's `name`                               |
| `dict[str, float]`    | one entry per key (e.g. `ArrayPRF1` writes `array_precision`/`_recall`/`_f1`) |
| `(value, extra)` tuple | the value, plus `.extra` structured detail (e.g. `schema_errors`) |
| `MetricResult`        | taken as-is (a float that already carries `.extra`)               |
| `None`                | the metric opts out for this node — no entry                      |

See [writing a custom metric](../metrics/custom-metric.md) for the full surface.

## See also

- [The evaluation model](evaluation-model.md) — the four-phase pipeline this idea
  lives inside, and the root → `report.score` wiring.
- [Aggregation and weighting](aggregation-and-weighting.md) — how object/array
  metrics combine children, and `weight_mode`.
- [Array alignment](array-alignment.md) — pairing elements before scoring them.
- [Writing a custom metric](../metrics/custom-metric.md) — implement the contract
  yourself.
- [`MeanScore`](../metrics/catalog/mean-score.md) — the default representative.
