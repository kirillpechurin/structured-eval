# Stage 11 (доп.) — Smoke test: pytest-style assertions

**Цель**: assert-методы на `EvalReport` для использования прямо в тестах. Контракт —
pytest-style: при провале `raise AssertionError` с внятным сообщением, при успехе `None`.

## Решения

- `assert_no_parse_errors()` — падает, если документ не распарсился.
- `assert_score(min)` — сначала проверяет parse; падает, если нет key-метрики или score < min.
- `assert_field(path, min)` — падает, если пути нет, у поля нет score, или score < min
  (в сообщении — `actual`/`expected`).
- `assert_metric(name, min)` — падает, если метрика не вычислялась (перечисляет доступные)
  или value < min.
- `assert_schema_valid()` — падает при `schema_validity == 0.0` или непустых `schema_errors`.

## Скрипт (проверено — проходит)

```python
from structured_eval import evaluate, EvalConfig, FieldConfig
from structured_eval.metrics import OverallScore, Coverage, SchemaValidity, ExactMatch

cfg = EvalConfig(
    metrics=[OverallScore(), Coverage()],
    key_metric=OverallScore(),
    fields={
        "vendor": FieldConfig(key_metric=ExactMatch()),
        "total":  FieldConfig(key_metric=ExactMatch()),
    },
)


def must_raise(fn, *needles):
    try:
        fn()
    except AssertionError as e:
        for n in needles:
            assert n in str(e), f"missing {n!r} in {e!r}"
        return
    raise AssertionError(f"expected AssertionError from {fn}")


# vendor wrong, total right → overall 0.5
r = evaluate({"vendor": "X", "total": 100}, {"vendor": "Acme", "total": 100}, cfg)

# ── passing assertions return None ───────────────────────────────────────────
assert r.assert_no_parse_errors() is None
assert r.assert_score(0.5) is None
assert r.assert_field("total", 1.0) is None
assert r.assert_metric("coverage", 1.0) is None

# ── failing assertions raise with informative messages ───────────────────────
must_raise(lambda: r.assert_score(0.9), "overall_score", "0.5", "0.9")
must_raise(lambda: r.assert_field("vendor", 1.0), "vendor", "'X'", "'Acme'")
must_raise(lambda: r.assert_field("missing", 1.0), "no field", "'missing'")
must_raise(lambda: r.assert_metric("nope", 1.0), "not computed", "coverage")
must_raise(lambda: r.assert_metric("coverage", 1.1), "coverage", "1.1")

# ── parse error short-circuits ───────────────────────────────────────────────
bad = evaluate("{ broken", {"vendor": "Acme"}, cfg)
must_raise(bad.assert_no_parse_errors, "parse error")
must_raise(lambda: bad.assert_score(0.0), "parse error")

# ── schema validity ──────────────────────────────────────────────────────────
schema = {"type": "object", "required": ["status"],
          "properties": {"status": {"type": "string"}}}
ok = evaluate({"status": "paid"}, {"status": "paid"}, EvalConfig(metrics=[SchemaValidity(schema)]))
assert ok.assert_schema_valid() is None
invalid = evaluate({"total": 1}, {"status": "paid"}, EvalConfig(metrics=[SchemaValidity(schema)]))
must_raise(invalid.assert_schema_valid, "schema invalid", "missing: status")

# ── no key metric → assert_score is unusable, assert_metric still works ───────
no_key = evaluate({"a": 1}, {"a": 1}, EvalConfig(metrics=[Coverage()]))
must_raise(lambda: no_key.assert_score(0.5), "no score available")
assert no_key.assert_metric("coverage", 1.0) is None

print("Stage 11 asserts smoke OK")
```

## Запуск

```bash
python -c "$(sed -n '/^```python$/,/^```$/p' tests/smoke/stage11_asserts.md | sed '1d;$d')"
# → Stage 11 asserts smoke OK
```
