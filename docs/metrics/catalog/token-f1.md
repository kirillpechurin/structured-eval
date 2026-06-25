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
word only helps as often as it appears on both sides.

## Parameters

None — `TokenF1()`.

## How it's computed

```text
tokens = lowercase → strip punctuation → split on whitespace
same   = number of shared tokens, counted with multiplicity  (Counter(a) & Counter(e))
precision = same / len(actual_tokens)
recall    = same / len(expected_tokens)
score     = 2 · precision · recall / (precision + recall)
```

Two empty texts score `1.0`; one empty (against a non-empty) scores `0.0`.

## Example

Word overlap earns partial credit — both of these miss one word out of three and score
`0.667`:

```python
from structured_eval import evaluate, EvalConfig, FieldConfig, TokenF1

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
float(report.field_scores["summary"].metrics["token_f1"])   # 0.667 — shares learn/python/basics
```

## Edge cases

- **Punctuation & case ignored** — `"hello, world."` and `"hello world"` tokenize the
  same, so they score `1.0`.
- **Repeated words count with multiplicity** — `"the the cat"` vs `"the cat"` is `0.8`,
  not `1.0` (the extra `the` has no partner). This is what makes it SQuAD F1 rather than
  a plain set overlap.
- **Articles are kept** — `a`/`an`/`the` are *not* stripped (unlike SQuAD's answer
  normalization); they count as tokens like any other word.
- **Empty handling** — both empty → `1.0`; exactly one empty → `0.0`.
- **Word order is ignored** — it's a bag of tokens. For order-sensitive or
  character-level similarity use [`Fuzzy`](fuzzy.md).

## See also

- [`Fuzzy`](fuzzy.md) — character/sequence similarity (order-sensitive variants).
- [`RegexMatch`](regex-match.md) — all-or-nothing after normalization.
- [`ExactMatch`](exact-match.md) — strict equality.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
