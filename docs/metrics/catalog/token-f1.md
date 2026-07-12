# TokenF1

|            |                       |
|------------|-----------------------|
| **Class**  | `TokenF1`             |
| **Key**    | `token_f1`            |
| **Branch** | field (scalar leaves) |
| **Needs**  | `expected`            |

## What it measures

A **graded** overlap of words between two texts, in `[0, 1]`. It splits each value into
tokens and scores the F1 of their overlap — so `"Intro to Python"` vs
`"Introduction to Python"` gets partial credit (the shared `to`/`python`) instead of a
flat `0.0`. Use it for free-text fields (titles, summaries, descriptions) where word
overlap matters more than exact wording.

This is the standard **SQuAD token F1**: tokens are matched as a multiset, so a repeated
word only helps as often as it appears on both sides. On its defaults the metric
reproduces the `f1_score` of the official SQuAD v1.1 evaluation script, normalization
included.

## Parameters

| Name                 | Type   | Default | Meaning                                     |
|----------------------|--------|---------|---------------------------------------------|
| `ignore_case`        | `bool` | `True`  | lowercase both sides before tokenizing      |
| `ignore_punctuation` | `bool` | `True`  | delete punctuation (Python's `string.punctuation`) |
| `ignore_articles`    | `bool` | `True`  | drop the standalone words `a`, `an`, `the`  |

Together the defaults are exactly SQuAD's `normalize_answer`. Turn a step off when what
it removes is meaningful in that field — codes, identifiers, formatted strings:

```python
TokenF1(ignore_case=False)         # "AB" vs "ab" scores below 1.0
TokenF1(ignore_punctuation=False)  # "fox." and "fox" are distinct tokens
TokenF1(ignore_articles=False)     # "the" counts as a token like any other
```

The toggles are independent: articles are dropped whether or not the text was
case-folded, so `TokenF1(ignore_case=False)` still treats `"The cat"` as `"cat"`.

Punctuation is **deleted**, not turned into a separator, so `"don't"` is the single
token `dont` — as in the reference script. With `ignore_punctuation=False` nothing is
removed and nothing new is split: punctuation simply stays attached to the token it
touches, making `"fox."` and `"fox"` different words.

## How it's computed

```text
tokens = lowercase → delete punctuation → drop articles → split on whitespace
same   = number of shared tokens, counted with multiplicity  (Counter(a) & Counter(e))
precision = same / len(actual_tokens)
recall    = same / len(expected_tokens)
score     = 2 · precision · recall / (precision + recall)
```

Two empty texts score `1.0`; one empty (against a non-empty) scores `0.0`. "Empty" here
means *after* normalization, so on the defaults `"the"` and `"!!!"` are empty too.

This is the reference algorithm with two deliberate departures, both because the metric
scores fields rather than question answers: two empty strings score `1.0` (the script
returns `0.0` — an empty answer is a failed answer), and a non-`str` value scores `0.0`
instead of being coerced.

## Example

Word overlap earns partial credit. `title` shares two words out of three on each side →
`0.667`; `summary` shares three out of four, because the article `the` is dropped before
counting → `0.75`:

```python
from structured_eval import evaluate
from structured_eval.models import EvalConfig, FieldConfig
from structured_eval.metrics import TokenF1

config = EvalConfig(fields={
    "title": FieldConfig(metrics=[TokenF1()]),
    "summary": FieldConfig(metrics=[TokenF1()]),
})
report = evaluate(
    {"title": "Intro to Python",        "summary": "Learn Python basics fast"},
    {"title": "Introduction to Python", "summary": "Learn the basics of Python"},
    config,
)

float(report.field_scores["title"].metrics["token_f1"])     # 0.667 — shares "to", "python"
float(report.field_scores["summary"].metrics["token_f1"])   # 0.75  — shares learn/python/basics
```

## Edge cases

- **Punctuation & case ignored** — `"hello, world."` and `"hello world"` tokenize the
  same, so they score `1.0`. So do `"don't"` and `"dont"`.
- **Repeated words count with multiplicity** — `"cat cat"` vs `"cat"` is `0.667`, not
  `1.0` (the extra `cat` has no partner). This is what makes it SQuAD F1 rather than a
  plain set overlap.
- **Articles are stripped** — `a`/`an`/`the` are dropped as standalone words, so
  `"a cat and an ox"` and `"cat and ox"` score `1.0`. Pass `ignore_articles=False` to
  count them as ordinary tokens.
- **Strings only** — if either side isn't a `str` the score is `0.0` (no coercion);
  `None` vs `"none"` is `0.0`, not a match.
- **Both `null` → `1.0`** — a null expectation met by a null value is a correct answer,
  not a type mismatch. Only both sides `None` count; one-sided `None` stays `0.0`.
- **Empty strings** — both empty → `1.0`,
  exactly one empty → `0.0` — where "empty" means empty after normalization, so `"the"`
  vs `""` is `1.0`.
- **Word order is ignored** — it's a bag of tokens. For order-sensitive or
  character-level similarity use [`Fuzzy`](fuzzy.md).

## See also

- [`Fuzzy`](fuzzy.md) — character/sequence similarity (order-sensitive variants).
- [`RegexMatch`](regex-match.md) — all-or-nothing after normalization.
- [`ExactMatch`](exact-match.md) — strict equality.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
