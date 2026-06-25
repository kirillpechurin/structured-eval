# RegexMatch

|            |                       |
|------------|-----------------------|
| **Class**  | `RegexMatch`          |
| **Key**    | `regex_match`         |
| **Branch** | field (scalar leaves) |
| **Needs**  | `expected`            |

## What it measures

String equality **after a configurable cleanup**: `1.0` when the rewritten strings
match, else `0.0`. By default it lowercases, trims, and collapses runs of whitespace —
so `"Beginner"` matches `"beginner"` and `"  Intro   to  Python "` matches
`"Intro to Python"`. Use it for text fields where casing and spacing (or any pattern
you choose) are noise, but the wording must otherwise be identical.

It is **string-only**: if either side is not a `str`, the score is `0.0` — values are
never coerced. For numbers use [`Numeric`](numeric.md); for verbatim equality of any
type use [`ExactMatch`](exact-match.md).

## Parameters

| Param     | Default   | Meaning                                                |
|-----------|-----------|--------------------------------------------------------|
| `pattern` | `r"\s+"`  | regex (str or compiled) matched after lower/strip      |
| `repl`    | `" "`     | replacement for each `pattern` match                   |
| `lower`   | `True`    | lowercase both sides before substituting               |
| `strip`   | `True`    | strip surrounding whitespace before and after          |

The defaults give "ignore casing and whitespace." Turn off `lower`/`strip` for a
case- or space-sensitive comparison, or change `pattern`/`repl` to ignore something
else (drop punctuation, dashes → spaces, …).

## How it's computed

```text
norm(v) = lower?(v) → strip?(v) → pattern.sub(repl, v) → strip?(v)
score   = 1.0 if both are str and norm(actual) == norm(expected) else 0.0
```

If either value isn't a string the metric short-circuits to `0.0` before any of this.

## Example

Casing and irregular spacing are ignored; the wording still has to match:

```python
from structured_eval import evaluate, EvalConfig, FieldConfig, RegexMatch

config = EvalConfig(fields={
    "level": FieldConfig(metrics=[RegexMatch()]),
    "title": FieldConfig(metrics=[RegexMatch()]),
})
report = evaluate(
    {"level": "Beginner", "title": "  Intro   to  Python "},
    {"level": "beginner", "title": "Intro to Python"},
    config,
)

float(report.field_scores["level"].metrics["regex_match"])   # 1.0 — casing only
float(report.field_scores["title"].metrics["regex_match"])   # 1.0 — extra spaces collapsed
```

Pass `pattern` / `repl` to ignore more — here, strip punctuation so `"Python: The
Basics!"` matches `"python the basics"`:

```python
from structured_eval import evaluate, EvalConfig, FieldConfig, RegexMatch

config = EvalConfig(fields={
    "title": FieldConfig(metrics=[RegexMatch(pattern=r"[^\w\s]", repl="")]),
})
report = evaluate(
    {"title": "Python: The Basics!"},
    {"title": "python the basics"},
    config,
)

float(report.field_scores["title"].metrics["regex_match"])   # 1.0 — punctuation dropped
```

## Edge cases

- **String-only** — non-`str` on either side scores `0.0`, never coerced. `12` vs
  `12.0` is `0.0` here (use [`Numeric`](numeric.md)), and `None` vs `"none"` is `0.0`
  (no accidental match on stringified `None`).
- **Default scope is narrow** — only case and whitespace. Accents and punctuation are
  **kept**: `"Café" ≠ "cafe"` and `"Hello!" ≠ "hello"` out of the box. Widen with
  `pattern` (e.g. `r"[^\w\s]"` to drop punctuation).
- **Flags off** — `RegexMatch(lower=False)` is case-sensitive; `RegexMatch(strip=False)`
  keeps surrounding whitespace.
- **Still all-or-nothing** — after the rewrite it's exact equality, no partial credit.
  For graded similarity use [`Fuzzy`](fuzzy.md) or [`TokenF1`](token-f1.md).

## See also

- [`ExactMatch`](exact-match.md) — equality with no rewrite, any type.
- [`Fuzzy`](fuzzy.md) / [`TokenF1`](token-f1.md) — graded string similarity.
- [`Numeric`](numeric.md) — for numeric fields.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
