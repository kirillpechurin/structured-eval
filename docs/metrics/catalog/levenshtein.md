# Levenshtein

|            |                                |
|------------|--------------------------------|
| **Class**  | `Levenshtein`                  |
| **Key**    | `levenshtein`                  |
| **Branch** | field (scalar leaves)          |
| **Needs**  | `expected`                     |

> Needs the optional `rapidfuzz` dependency: `pip install 'structured-eval[fuzzy]'`.

## What it measures

A **graded** edit-distance similarity in `[0, 1]`: the normalized Levenshtein ratio,
where `1.0` is identical and the score drops as more single-character edits (insert,
delete, substitute) are needed. Good for catching small typos in codes, ids, and short
strings, where word order isn't a factor.

It's a thin alias for [`Fuzzy(method="ratio")`](fuzzy.md) — same arithmetic, more
discoverable name.

## Parameters

| Param               | Default   | Meaning                                          |
|---------------------|-----------|--------------------------------------------------|
| `ignore_case`       | `True`    | lowercase both sides before comparing            |
| `ignore_whitespace` | `True`    | strip surrounding whitespace before comparing    |

(It also accepts `method`, but the point of `Levenshtein` is the ratio; for other
RapidFuzz scorers use [`Fuzzy`](fuzzy.md) directly.)

## How it's computed

```text
score = rapidfuzz.fuzz.ratio(actual, expected) / 100        # string-only; non-str → 0.0
```

`ratio` is the normalized Levenshtein similarity — it's order-sensitive (unlike
`Fuzzy`'s default `token_sort_ratio`).

## Example

A one-character OCR-style slip scores high but not perfect:

```python
from structured_eval import evaluate
from structured_eval.models import EvalConfig, FieldConfig
from structured_eval.metrics import Levenshtein

config = EvalConfig(fields={"code": FieldConfig(metrics=[Levenshtein()])})
report = evaluate({"code": "COURSE-1O1"}, {"code": "COURSE-101"}, config)

float(report.field_scores["code"].metrics["levenshtein"])   # 0.9 — one char off (O vs 0)
```

```python
from structured_eval.metrics import Levenshtein

Levenshtein().score("kitten", "sitting")   # 0.615 — three edits over 13 chars
```

## Edge cases

- **Strings only** — non-`str` on either side scores `0.0` (inherited from
  [`Fuzzy`](fuzzy.md)), except two `None`s, which agree → `1.0`.
- **Order-sensitive** — reordered words score low; use
  [`Fuzzy(method="token_sort_ratio")`](fuzzy.md) or [`TokenF1`](token-f1.md) for
  order-insensitive comparison.
- **Independent normalization** — `ignore_case=False` keeps case,
  `ignore_whitespace=False` keeps surrounding whitespace; the two toggle
  separately (inherited from [`Fuzzy`](fuzzy.md)).
- **Optional dependency** — raises `ImportError` (with install hint) without
  `rapidfuzz`.

## See also

- [`Fuzzy`](fuzzy.md) — the general fuzzy metric this aliases.
- [`TokenF1`](token-f1.md) — token-overlap F1 (order-insensitive).
- [The metric catalog](../index.md) — all metrics and the return-shape model.
