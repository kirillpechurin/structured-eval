# Fuzzy

|            |                       |
|------------|-----------------------|
| **Class**  | `Fuzzy`               |
| **Key**    | `fuzzy`               |
| **Branch** | field (scalar leaves) |
| **Needs**  | `expected`            |

> Needs the optional `rapidfuzz` dependency: `pip install 'structured-eval[fuzzy]'`.

## What it measures

A **graded** string similarity in `[0, 1]`, powered by [RapidFuzz](https://github.com/maxbachmann/RapidFuzz).
It rewards near-misses — typos, abbreviations, reordered words — so `"Sara Jonson"` vs
`"Sarah Johnson"` scores high instead of `0.0`. Use it for free-text fields where you
want character/word-level closeness rather than exact equality.

Pick the comparison style with `method`; by default it's order-insensitive
(`token_sort_ratio`).

## Parameters

| Param       | Default              | Meaning                                            |
|-------------|----------------------|----------------------------------------------------|
| `method`    | `"token_sort_ratio"` | which RapidFuzz scorer to use (see below)          |
| `normalize` | `True`               | strip + lowercase both sides before comparing      |

`method` options (`FuzzyMethod`):

| Method              | Behavior                                              |
|---------------------|-------------------------------------------------------|
| `ratio`             | plain Levenshtein ratio (order-sensitive)             |
| `partial_ratio`     | best matching substring (good for "contains")         |
| `token_sort_ratio`  | sorts tokens first — ignores word order (default)     |
| `token_set_ratio`   | set of tokens — also ignores duplicate/extra words    |

## How it's computed

```text
a, e = actual, expected            # string-only; non-str → 0.0
if normalize: a, e = lower(strip(a)), lower(strip(e))
score = rapidfuzz_scorer(a, e) / 100
```

## Example

A typo'd instructor name still scores high, and the default ignores word order:

```python
from structured_eval import evaluate
from structured_eval.models import EvalConfig, FieldConfig
from structured_eval.metrics import Fuzzy

config = EvalConfig(fields={
    "instructor_name": FieldConfig(metrics=[Fuzzy()]),
    "title": FieldConfig(metrics=[Fuzzy()]),
})
report = evaluate(
    {"instructor_name": "Sara Jonson",  "title": "Python Programming Intro"},
    {"instructor_name": "Sarah Johnson", "title": "Intro Python Programming"},
    config,
)

float(report.field_scores["instructor_name"].metrics["fuzzy"])   # 0.917 — two small typos
float(report.field_scores["title"].metrics["fuzzy"])             # 1.0 — same words, reordered
```

Switch `method` for a different notion of similarity — `ratio` is order-sensitive,
`partial_ratio` rewards a substring match:

```python
from structured_eval.metrics import Fuzzy

Fuzzy(method="ratio").score("Python Programming Intro", "Intro Python Programming")
# 0.75 — plain ratio is hurt by the reordering

Fuzzy(method="partial_ratio").score("Python", "Intro to Python Programming")
# 1.0 — "Python" appears verbatim inside the longer string
```

## Edge cases

- **Strings only** — if either side isn't a `str` the score is `0.0` (no coercion);
  `None` vs `"none"` is `0.0`, numbers like `123` vs `123.0` are `0.0`. Score numbers
  with [`Numeric`](numeric.md) / [`NumericCloseness`](numeric-closeness.md).
- **`normalize=False`** — keeps case and surrounding whitespace, so `"ACME"` vs
  `"acme"` drops below `1.0`.
- **Order sensitivity depends on `method`** — `token_sort_ratio` / `token_set_ratio`
  ignore word order; `ratio` does not.
- **Optional dependency** — without `rapidfuzz` installed, using `Fuzzy` raises
  `ImportError` with the install hint.

## See also

- [`Levenshtein`](levenshtein.md) — thin alias for `Fuzzy(method="ratio")`.
- [`TokenF1`](token-f1.md) — token-overlap F1 (bag of words).
- [`RegexMatch`](regex-match.md) — all-or-nothing after normalization.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
