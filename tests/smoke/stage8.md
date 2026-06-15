# Stage 8 — Smoke test: root-метрики

**Цель этапа**: метрики уровня документа (`RootMetric`, срабатывают только на корне):
`OverallScore` (взвешенная агрегация по дереву — headline), `SchemaValidity` (валидация
по схеме), `Coverage` (полнота).

## Решения этапа

- **Именование**: разрешён открытый вопрос — ключ метрики **`schema_validity`** (совпадает
  с именем класса `SchemaValidity`; вариант `schema_valid` из ранних RFC отвергнут).
- **`OverallScore`** — взвешенное среднее match-критерия по листьям (`weight` из `FieldConfig`,
  хранится на `EvalNode.weight`, проставляется в match-фазе). Отсутствующие листья → 0.
- **`Coverage`** — доля ожидаемых листьев, присутствующих (non-null) в actual; корректность
  значения не учитывается. Элементы массива, пропущенные при выравнивании, считаются
  array-метриками, не здесь.
- **`SchemaValidity`** валидирует `node.actual` через `schema/validator.validate` (Pydantic-
  модель или JSON Schema dict); ошибки складываются в `self.schema_errors`, движок переносит
  их в `report.schema_errors`.

## Скрипт (проверено — проходит)

```python
from structured_eval import evaluate, EvalConfig, FieldConfig
from structured_eval.metrics import OverallScore, Coverage, SchemaValidity, ExactMatch

# ── OverallScore: взвешенное среднее (vendor неверный w=3, total верный w=1) → 0.25 ──
cfg = EvalConfig(
    metrics=[OverallScore(), Coverage()],
    key_metric=OverallScore(),
    fields={
        "vendor": FieldConfig(key_metric=ExactMatch(), weight=3.0),
        "total":  FieldConfig(key_metric=ExactMatch(), weight=1.0),
    },
)
r = evaluate({"vendor": "X", "total": 100}, {"vendor": "Acme", "total": 100}, cfg)
assert r.metrics["overall_score"] == 0.25
assert r.score == 0.25 and r.score_label == "overall_score"
assert r.metrics["coverage"] == 1.0   # оба поля присутствуют

# ── Coverage: отсутствующее ожидаемое поле → 1/2 ─────────────────────────────
r2 = evaluate({"a": 1}, {"a": 1, "b": 2}, EvalConfig(metrics=[Coverage()]))
assert r2.metrics["coverage"] == 0.5

# ── SchemaValidity (JSON Schema dict) ────────────────────────────────────────
schema = {"type": "object", "required": ["status"],
          "properties": {"status": {"type": "string"}}}
ok = evaluate({"status": "paid"}, {"status": "paid"}, EvalConfig(metrics=[SchemaValidity(schema)]))
assert ok.metrics["schema_validity"] == 1.0 and ok.schema_errors == []

bad = evaluate({"total": 1}, {"status": "paid"}, EvalConfig(metrics=[SchemaValidity(schema)]))
assert bad.metrics["schema_validity"] == 0.0
assert bad.schema_errors == ["missing: status"]

print("Stage 8 smoke OK")
```

## Запуск

```bash
python -c "$(sed -n '/^```python$/,/^```$/p' tests/smoke/stage8.md | sed '1d;$d')"
# → Stage 8 smoke OK
```
