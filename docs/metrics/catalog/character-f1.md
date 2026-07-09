# CharacterF1

|            |                       |
|------------|-----------------------|
| **Class**  | `CharacterF1`         |
| **Key**    | `character_f1`        |
| **Branch** | field (scalar leaves) |
| **Needs**  | `expected`            |

## What it measures

Character-level **F1** between two short strings: how much their character *multisets*
overlap. It rewards near-misses that share most of their letters — typos, spacing,
casing, light punctuation — where [`ExactMatch`](exact-match.md) would score `0.0`.
Good for short free-text fields (names, codes, single words) where token-level
[`TokenF1`](token-f1.md) is too coarse and edit-distance [`Fuzzy`](fuzzy.md) needs an
extra dependency.

## Parameters

| Name                 | Type   | Default | Meaning                                            |
|----------------------|--------|---------|----------------------------------------------------|
| `ignore_case`        | `bool` | `True`  | lowercase both sides before comparing              |
| `ignore_whitespace`  | `bool` | `True`  | drop all whitespace characters                     |
| `ignore_punctuation` | `bool` | `True`  | delete punctuation (Python's `string.punctuation`)  |

The defaults apply all three normalizations. Turn one off when the characters it
removes are semantically significant — codes, identifiers, formatted strings:

```python
CharacterF1(ignore_case=False)         # "AB" vs "ab" scores below 1.0
CharacterF1(ignore_punctuation=False)  # "," and "." count toward the multiset
CharacterF1(ignore_whitespace=False)   # spaces count toward the multiset
```

## How it's computed

Both sides are normalized (by default: lowercased, stripped of punctuation and
whitespace) and reduced to a multiset of characters (a `Counter`). The shared count is
the multiset intersection:

```text
same = Σ min(count_a(c), count_e(c))
precision = same / len(a)
recall    = same / len(e)
score = 2 · precision · recall / (precision + recall)
```

Because it's a multiset, a repeated character only counts as many times as it appears on
**both** sides.

## Example

```python
from structured_eval import evaluate
from structured_eval.models import EvalConfig, FieldConfig
from structured_eval.metrics import CharacterF1

config = EvalConfig(fields={"name": FieldConfig(metrics=[CharacterF1()])})

# "kitten" vs "sitting" share {i,t,t,n} → p=4/6, r=4/7 → F1 ≈ 0.615
report = evaluate({"name": "kitten"}, {"name": "sitting"}, config)
report.field_scores["name"].metrics["character_f1"]   # 0.615
```

## Edge cases

- **String only** — if either side isn't a `str` the score is `0.0` (no coercion); that
  includes `None` and numbers.
- **Both empty → `1.0`** — two empty (or punctuation-only) strings match vacuously.
- **`ignore_punctuation=False`** — punctuation is kept, so `"!!!"` is no longer an empty
  string and is compared like any other.
- **ASCII punctuation only** — the set is Python's `string.punctuation`, the same one
  [`TokenF1`](token-f1.md) uses. So `_` *is* dropped (`"a_b"` vs `"ab"` → `1.0`), while
  non-ASCII punctuation such as `«»—` is **not** and counts toward the multiset.
- **One side empty → `0.0`** — including a string that normalizes to empty.
- **Order-blind** — `"abc"` and `"cba"` score `1.0` (it's a multiset, not a sequence).
  Use [`Levenshtein`](levenshtein.md) when order matters.

## See also

- [`TokenF1`](token-f1.md) — the same idea at the *token* level (whole words).
- [`Fuzzy`](fuzzy.md) / [`Levenshtein`](levenshtein.md) — edit-distance similarity (rapidfuzz).
- [The metric catalog](../index.md) — all metrics and the return-shape model.
