# Array alignment

Objects match by field name — `instructor.name` lines up with `instructor.name`.
Arrays have no such names: before you can score `["mix", "bake"]` against
`["bake", "mix"]`, you must decide *which actual element answers which expected
one*. That decision is **alignment**, and it's the one job arrays add on top of
the rest of the model.

Alignment is purely **structural** — it produces a set of pairs, not scores. The
[value-aware array metrics](../metrics/index.md) grade those pairs afterwards.
Keep the two stages separate in your head: *align, then score*.

## The result: matched / missed / spurious

Every strategy returns the same `ArrayMatchResult`, exposed at
`report.array_matches[path]`:

| Field      | Meaning                                              | Confusion role |
|------------|------------------------------------------------------|----------------|
| `matched`  | `(expected_idx, actual_idx)` pairs that were aligned | candidate TP   |
| `missed`   | expected indices with no actual counterpart          | FN             |
| `spurious` | actual indices absent from expected                  | FP             |

```python
from structured_eval import evaluate
from structured_eval.models import ArrayFieldConfig, EvalConfig
from structured_eval.metrics import ArrayPRF1

cfg = EvalConfig(fields={"steps": ArrayFieldConfig(metrics=[ArrayPRF1()])})
report = evaluate({"steps": ["mix", "bake", "cool"]}, {"steps": ["mix", "bake"]}, cfg)

m = report.array_matches["steps"]
m.matched    # [(0, 0), (1, 1)]
m.missed     # []
m.spurious   # [2]   — the extra "cool" has no expected counterpart
```

> `matched` is only a *candidate* TP: a pair still has to be graded by an array
> metric. Whether a matched-but-imperfect pair counts depends on that metric's
> `threshold`/`mode` — see [array metrics](../metrics/index.md). `array_matches`
> itself is counts-of-shape only.

## Strategies

Pick the strategy on the field's `ArrayFieldConfig`; tune it with `params`.

| Strategy             | Pairs by                         | Use when                                     |
|----------------------|----------------------------------|----------------------------------------------|
| `by_index` (default) | position (i-th ↔ i-th)           | order is meaningful (steps, rankings)        |
| `by_key`             | one or more key fields, greedily best-first | order is *not* meaningful (sets, rows by id) |
| `hungarian`          | globally optimal one-to-one      | order-free + you want the best overall pairing |

### `by_index` — positional

The i-th expected pairs with the i-th actual; the tail of the longer list becomes
`missed` or `spurious`. No comparison happens during alignment. This is the
default, and the right choice for positionally significant lists — the `steps`
snippet above is `by_index`: `matched [(0, 0), (1, 1)]`, `spurious [2]`.

### `by_key` — match on a key, greedily best-first

Extracts a key from each element (the `key` field, or the whole element when
`key` is `None`), scores keys with `key_metric` (default `ExactMatch`), and pairs
those clearing `threshold`. Order no longer matters:

```python
from structured_eval import evaluate
from structured_eval.models import ArrayFieldConfig, ArrayStrategy, EvalConfig
from structured_eval.metrics import ArrayPRF1

cfg = EvalConfig(fields={"items": ArrayFieldConfig(
    strategy=ArrayStrategy.BY_KEY,
    params={"key": "sku"},
    metrics=[ArrayPRF1()],
)})
report = evaluate(
    {"items": [{"sku": "B", "qty": 5}, {"sku": "A", "qty": 2}]},
    {"items": [{"sku": "A", "qty": 2}, {"sku": "B", "qty": 3}]},
    cfg,
)
report.array_matches["items"].matched   # [(0, 1), (1, 0)] — A↔A, B↔B despite the order
```

`params` for `by_key`: `key`, `key_metric`, `threshold` (default `1.0`).

`key` may also name **several fields** — a composite key, for records identified
by a combination such as `(sku, warehouse)`. Each field is scored with
`key_metric` and the element's key score is the mean over the fields, so with the
defaults every field must match and elements sharing one field no longer collide:

```python
cfg = EvalConfig(fields={"items": ArrayFieldConfig(
    strategy=ArrayStrategy.BY_KEY,
    params={"key": ["sku", "warehouse"]},
    metrics=[ArrayPRF1()],
)})
report = evaluate(
    {"items": [{"sku": "A", "warehouse": "west", "qty": 2},
               {"sku": "A", "warehouse": "east", "qty": 1}]},
    {"items": [{"sku": "A", "warehouse": "east", "qty": 1},
               {"sku": "A", "warehouse": "west", "qty": 2}]},
    cfg,
)
report.array_matches["items"].matched   # [(0, 1), (1, 0)] — same sku, kept apart by warehouse
```

Because the fields are averaged, a soft `key_metric` lets a strong field carry a
weaker one (an exact `sku` plus a fuzzy `name` scoring `0.6` means `0.8`). A
one-field key (`["sku"]`) is the mean of one score — identical to `"sku"`.

Pairing is **globally greedy**: every threshold-clearing pair is ranked by key
score (highest first) and claimed one-to-one. So a *soft* key (a fuzzy/numeric
metric) picks the strongest available partner rather than the first one found,
and the outcome doesn't depend on element order. With an exact key (all passing
scores tie at `1.0`) it reduces to plain first-match.

```python
# soft key: pair people by a fuzzy name match, threshold 0.6
cfg = EvalConfig(fields={"people": ArrayFieldConfig(
    strategy=ArrayStrategy.BY_KEY,
    params={"key": "name", "key_metric": "fuzzy", "threshold": 0.6},
    metrics=[ArrayPRF1()],
)})
report = evaluate(
    {"people": [{"name": "Jonathan"}, {"name": "Bob"}]},
    {"people": [{"name": "John"}, {"name": "Bobby"}]},
    cfg,
)
report.array_matches["people"].matched   # [(0, 0), (1, 1)] — Jonathan↔John, Bob↔Bobby
```

### `hungarian` — globally optimal one-to-one

Builds a similarity matrix `S[i,j] = score(expected[i], actual[j])` and solves the
optimal assignment (`scipy.optimize.linear_sum_assignment`). Unlike `by_key`'s
greedy approximation, it's the provably best one-to-one pairing for the whole
array. A pair counts as matched only when its similarity clears `threshold`
(default `0.8`). Requires the `align` extra (`pip install 'structured-eval[align]'`).

The element similarity is `scorer`:

- a single `Scorer` (a metric, its name, or an `(actual, expected) -> float`
  callable) — applied to the whole element;
- a `dict[str, Scorer]` — per-field scorers for arrays of objects; the element
  score is the mean over the union of fields;
- `None` — a type-aware default: numbers are graded by closeness, everything else
  (strings included) is exact; objects are scored field-by-field. Strings are not
  graded by default — ask for it with `"fuzzy"`.

```python
# pair invoice line items that are reordered AND fuzzily worded
cfg = EvalConfig(fields={"line_items": ArrayFieldConfig(
    strategy=ArrayStrategy.HUNGARIAN,
    params={
        "scorer": {"desc": "fuzzy", "price": "numeric_closeness"},
        "threshold": 0.7,
    },
)})
report = evaluate(
    {"line_items": [{"desc": "USB cable",    "price": 9.0},
                    {"desc": "HDMI adapter", "price": 19.0}]},
    {"line_items": [{"desc": "HDMI adaptor", "price": 20.0},
                    {"desc": "USB-C cable",  "price": 9.0}]},
    cfg,
)
report.array_matches["line_items"].matched   # [(0, 1), (1, 0)] — cable↔cable, HDMI↔HDMI
```

`params` for `hungarian`: `scorer`, `threshold` (default `0.8`), `key`.

`key` narrows what is compared: instead of the whole element, only the named
field — or **several fields**, a composite key such as `["sku", "warehouse"]`.
Field paths may be nested (`"who.first"`). `key` picks *what* is compared and
`scorer` *how*: with a `key` set, a `dict` scorer binds a scorer per key field,
a single scorer applies to each of them, and the element score is the mean over
the key fields. Naming a field that isn't in `key` is a `ValueError` — it would
otherwise be silently ignored:

```python
cfg = EvalConfig(fields={"items": ArrayFieldConfig(
    strategy=ArrayStrategy.HUNGARIAN,
    params={
        "key": ["sku", "warehouse"],       # match on the pair, ignore qty
        "scorer": {"sku": "fuzzy"},        # warehouse falls back to its type default
        "threshold": 0.9,
    },
)})
```

A one-field key (`["sku"]`) scores exactly like `"sku"`.

> The `scorer` here is the *alignment* similarity — it decides **who pairs with
> whom**, not the final element scores. Grading the paired elements is a separate
> step done by the array metric and the element's own metrics (see
> [from alignment to a score](#from-alignment-to-a-score)). Configure both if you
> want a soft pairing *and* graded credit.

> `by_key` vs `hungarian`: both pair order-independently, and both take a
> composite key. `by_key` is a cheap, scipy-free greedy approximation that scores
> every key field with the same `key_metric`; `hungarian` is the optimal
> assignment over a full similarity (whole element, or per-field with a scorer
> per key field), at the cost of the scipy dependency and an O(n²) matrix. Reach
> for `hungarian` when the key fields need different metrics, or when you want
> the globally best pairing rather than a greedy one.

## From alignment to a score

`array_matches` is the structural skeleton; the number comes from an array metric
reading it:

- [`ArrayPRF1`](../metrics/catalog/array-prf1.md) / `ArrayPrecision` / `ArrayRecall`
  / `ArrayF1` — slot-filling P/R/F1 over matched/missed/spurious.
- [`ArrayAccuracy`](../metrics/catalog/array-accuracy.md) — recall-flavored mean of
  matched-element scores (extra elements not penalized).
- [`ArrayCardinality`](../metrics/catalog/array-cardinality.md) — count agreement
  only.

A matched pair isn't automatically a true positive — the array metric grades each
matched element (by its representative) against its `threshold`/`mode`. So
alignment decides *who pairs with whom*; the metric decides *how good each pair
is*. The two knobs are independent: a generous alignment `scorer` can pair
elements that a strict `ArrayPRF1` (`mode="hard"`, `threshold=1.0`) still counts
as misses — lower the metric threshold or use `mode="soft"` for partial credit.

## Edge cases

- **Empty / one-sided arrays** — an empty actual makes every expected `missed`; an
  empty expected makes every actual `spurious`; both empty → vacuously aligned.
- **Key absent / non-dict element** (`by_key`, `hungarian`) — a missing key reads
  as `None`; an element that isn't an object can't yield a named key, so it won't
  pair on one.
- **Large arrays** (`hungarian`) — the similarity matrix is O(n²); beyond ~10 000
  cells it warns about cost.
- **Scalar-root arrays aren't navigable yet** — test an array as a *nested field*
  (a root-level list is on the roadmap).

## See also

- [The evaluation model](evaluation-model.md) — where arrays sit in the node tree.
- [Comparison is a metric](comparison-is-a-metric.md) — alignment scores keys with
  a metric, like everything else.
- [Array metrics](../metrics/index.md) — turning matched/missed/spurious into P/R/F1.
- [`ArrayPRF1`](../metrics/catalog/array-prf1.md) ·
  [`ArrayAccuracy`](../metrics/catalog/array-accuracy.md) — the common choices.
