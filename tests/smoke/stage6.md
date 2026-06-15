# Stage 6 — Smoke test: трёхфазный движок + `evaluate()` (dict)

**Цель этапа**: первый рабочий end-to-end `evaluate()` для dict-документов (массивы —
Stage 7). Три фазы: **match** (`engine/match.py` — строит дерево `ObjectNode`/`ScalarNode`
и считает field-метрики на листьях), **compute** (`engine/compute.py` — `apply_metric`
dispatch для object/root-метрик по дереву), **flatten** (`engine/report_builder.py` —
дерево → `EvalReport`).

## Решения этапа

- **Field-метрики считаются в фазе match** — сравнение листа и есть «матчинг» поля
  (v3). Фаза compute применяет только агрегирующие (object/root) метрики из `config.metrics`.
- **`config.key_metric`** добавляется к вычисляемым метрикам автоматически; его значение
  на корне → `report.score` (+ `score_label`).
- **`report.metrics`** = метрики корневого узла; вложенные объекты несут свои метрики
  в `field_scores[path].metrics`.
- **`ScalarNode.score`/`threshold`** в `FieldScore` — значение match-критерия поля
  (через `_match_criterion`), используется в `failed_fields()`.
- **`ExtraKeysPolicy`**: `PENALIZE` → лишние ключи идут в `spurious` (роняют precision);
  `IGNORE` (дефолт) → не штрафуют, но дают warning `[EXTRA_KEY]`. `[MISSING_FIELD]` —
  всегда. Список — без исключений.
- **Массив** пока: warning `[ARRAY_UNSUPPORTED]` + сравнение как скаляр.

## Скрипт (проверено — проходит)

```python
from structured_eval import evaluate, EvalConfig, FieldConfig
from structured_eval.core.config import ExtraKeysPolicy
from structured_eval.metrics import (
    ObjectPRF1, ObjectF1, ExactMatch, NormalizedMatch, Numeric, TokenF1, TypeMatch,
)

# ── Базовый сценарий: per-field метрики + key_metric → score ─────────────────
cfg = EvalConfig(
    metrics=[ObjectPRF1()],
    key_metric=ObjectF1(),
    fields={
        "vendor": FieldConfig(metrics=[NormalizedMatch()], key_metric=NormalizedMatch()),
        "total":  FieldConfig(metrics=[Numeric(tolerance=0.01)], key_metric=Numeric(tolerance=0.01)),
        "status": FieldConfig(key_metric=ExactMatch()),
    },
)
r = evaluate(
    actual={"vendor": "Acme Corp.", "total": 100.0, "status": "paid"},
    expected={"vendor": "acme  corp.", "total": 100.0, "status": "paid"},
    config=cfg,
)
assert r.parse_error is False
assert r.score_label == "object_f1" and r.score == 1.0
assert r.metrics == {"object_precision": 1.0, "object_recall": 1.0, "object_f1": 1.0}
assert r.field_scores["vendor"].metrics == {"normalized_match": 1.0}
assert r.field_scores["vendor"].score == 1.0
assert sorted(r.field_scores) == ["$", "status", "total", "vendor"]
assert r.warnings == []

# ── Missing + spurious (PENALIZE) → P=R=F1=2/3 ───────────────────────────────
r2 = evaluate(
    {"vendor": "Acme", "total": 100, "note": "x"},
    {"vendor": "Acme", "total": 100, "status": "paid"},
    EvalConfig(metrics=[ObjectPRF1()], extra_keys=ExtraKeysPolicy.PENALIZE),
)
assert abs(r2.metrics["object_f1"] - 2 / 3) < 1e-9
assert r2.metrics["object_precision"] == r2.metrics["object_recall"]
assert r2.warnings == ["[MISSING_FIELD] 'status' absent in actual"]

# ── Parse error и JSON-строка на входе ───────────────────────────────────────
bad = evaluate({"a": 1}, "{bad json", EvalConfig(metrics=[ObjectF1()]))
assert bad.parse_error is True and bad.parse_error_message
ok = evaluate('{"a": 1}', '{"a": 1}', EvalConfig(metrics=[ObjectF1()]))
assert ok.metrics["object_f1"] == 1.0

# ── Вложенный объект + несколько метрик на поле + failed_fields ──────────────
cfg2 = EvalConfig(metrics=[ObjectF1()], default_metrics=[ExactMatch(), TokenF1(), TypeMatch()])
r3 = evaluate(
    {"vendor": {"name": "Acme", "inn": "123"}},
    {"vendor": {"name": "Acme Corp", "inn": "123"}},
    cfg2,
)
assert sorted(r3.field_scores) == ["$", "vendor", "vendor.inn", "vendor.name"]
m = r3.field_scores["vendor.name"].metrics
assert m["exact_match"] == 0.0 and abs(m["token_f1"] - 2 / 3) < 1e-9 and m["type_match"] == 1.0
assert r3.field_scores["vendor"].metrics["object_f1"] == 0.5   # inn совпал, name — нет
assert [fs.path for fs in r3.failed_fields()] == ["vendor.name"]

print("Stage 6 smoke OK")
```

## Запуск

```bash
python -c "$(sed -n '/^```python$/,/^```$/p' tests/smoke/stage6.md | sed '1d;$d')"
# → Stage 6 smoke OK
```
