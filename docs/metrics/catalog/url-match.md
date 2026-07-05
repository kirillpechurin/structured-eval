# UrlMatch

|            |                       |
|------------|-----------------------|
| **Class**  | `UrlMatch`            |
| **Key**    | `url_match`           |
| **Branch** | field (scalar leaves) |
| **Needs**  | `expected`            |

## What it measures

A **binary** URL equivalence verdict (`1.0` / `0.0`). It compares two URLs by their
meaningful components rather than raw string equality, so cosmetically different but
equivalent URLs score as matching. Comparing extracted URL fields with
[`ExactMatch`](exact-match.md) or a text metric misclassifies equivalent URLs — trailing
slashes, scheme/host casing, `www.` prefixes, percent-encoding and query-parameter order
all differ without changing the target. `UrlMatch` normalizes those away before comparing.
(Ports are compared as-is — `:443` is significant.)

## Parameters

- `ignore_query` (bool, default `False`) — drop the query string entirely before comparing.
- `ignore_fragment` (bool, default `True`) — drop the `#fragment` before comparing.
- `ignore_www` (bool, default `True`) — strip a leading `www.` from the host.

## How it's computed

```text
normalize(url):
    lowercase scheme and host
    strip leading "www." from host            (unless ignore_www=False)
    percent-decode the path; drop a trailing "/"
    sort query params by (key, value)          (dropped if ignore_query)
    drop the fragment                          (unless ignore_fragment=False)

score = 1.0 if normalize(actual) == normalize(expected) else 0.0
```

Both sides must be non-empty strings that parse to a URL with a scheme **and** a host.

## Example

Casing, trailing slash and query-parameter order are normalized, so equivalent URLs match:

```python
from structured_eval import evaluate
from structured_eval.models import EvalConfig, FieldConfig
from structured_eval.metrics import UrlMatch

config = EvalConfig(fields={
    "homepage": FieldConfig(metrics=[UrlMatch()]),
    "enroll_url": FieldConfig(metrics=[UrlMatch()]),
})
report = evaluate(
    {"homepage": "https://Example.com",
     "enroll_url": "https://example.com/course?ref=a&id=7"},
    {"homepage": "https://example.com/",
     "enroll_url": "https://example.com/course?id=7&ref=a"},
    config,
)

float(report.field_scores["homepage"].metrics["url_match"])     # 1.0 — case + trailing slash
float(report.field_scores["enroll_url"].metrics["url_match"])   # 1.0 — query order ignored
```

## Edge cases

- **Strings only** — a non-string, an empty string, or `None` scores `0.0`.
- **Must be an absolute URL** — a bare path (`"/course"`) or plain text with no scheme/host
  is unparseable → `0.0`.
- **Different scheme/host/path → `0.0`** — `http://` vs `https://`, different hosts, and
  different paths are all non-matches.
- **Ports are compared as-is** — `:443` and `:8080` are kept, so
  `https://example.com:443` does not match `https://example.com`.
- **Query values matter** — only parameter *order* is normalized; `?a=1` vs `?a=2` is a
  non-match unless `ignore_query=True`.

## See also

- [`ExactMatch`](exact-match.md) — strict string equality without normalization.
- [`RegexMatch`](regex-match.md) — string equality after an optional regex rewrite.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
