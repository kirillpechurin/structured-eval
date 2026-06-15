# Stage 7 — Smoke test: массивы (выравнивание + array-метрики)

**Цель этапа**: поддержка массивов. Выравнивание — пакет `align/`: `BY_INDEX` (по
позиции) и обобщённый `BY_KEY` (ключ + метрика сравнения + порог, поглощает value/
similarity). Array-метрики (`ArrayPrecision/Recall/F1/PRF1`, `ArrayAccuracy`,
`ArrayCardinality`) считают качество как множества элементов.

## Решения этапа

- **Двойной путь.** `EvalNode.expected_path` расходится с `path` для элементов, выровненных
  не по позиции (`expected[1]` ↔ `actual[0]`): каждая сторона навигируется по своему индексу.
- **P/R/F1 с гейтингом по корректности** (как у объектов): выровненная пара — TP только если
  `element_score` ≥ `threshold` (hard) или вносит свой score дробно (soft). Поэтому
  выровненный-но-неверный элемент роняет precision и recall.
- **`element_score`** рекурсивен: scalar → match-критерий, object → `ObjectAccuracy`,
  array → `ArrayAccuracy`. Арифметика P/R/F1 переиспользует `_object_stats`.
- **`ArrayMatchResult`** (matched/missed/spurious по индексам) попадает в `report.array_matches`.
- Greedy-назначение для `BY_KEY` (optimal/Hungarian — позже).

## Скрипт (проверено — проходит)

```python
from structured_eval import evaluate, EvalConfig, ArrayFieldConfig, ObjectFieldConfig, FieldConfig
from structured_eval.core.config import ArrayStrategy
from structured_eval.metrics import (
    ArrayPRF1, ArrayF1, ArrayAccuracy, ArrayCardinality, NormalizedMatch, Numeric,
)

def items_cfg(item):
    return EvalConfig(
        metrics=[ArrayPRF1(), ArrayCardinality()],
        fields={"items": ArrayFieldConfig(strategy=ArrayStrategy.BY_KEY, key="id", item=item)},
    )

obj_item = ObjectFieldConfig(fields={
    "name":   FieldConfig(key_metric=NormalizedMatch()),
    "amount": FieldConfig(key_metric=Numeric()),
})

# ── BY_KEY с переупорядочиванием, идентичный контент → всё 1.0 ────────────────
r = evaluate(
    {"items": [{"id": "A", "name": "Widget", "amount": 10.0},
               {"id": "B", "name": "Gadget", "amount": 20.0}]},
    {"items": [{"id": "B", "name": "Gadget", "amount": 20.0},
               {"id": "A", "name": "Widget", "amount": 10.0}]},
    items_cfg(obj_item),
)
am = r.array_matches["items"]
assert am.matched == [(0, 1), (1, 0)] and am.missed == [] and am.spurious == []
assert r.field_scores["items"].metrics["array_f1"] == 1.0
assert r.field_scores["items"].metrics["array_cardinality"] == 1.0
assert r.warnings == []

# ── BY_KEY: лишний (X) + отсутствующий (B) → P=R=F1=0.5 ──────────────────────
r2 = evaluate(
    {"items": [{"id": "A", "name": "Widget"}, {"id": "X", "name": "Junk"}]},
    {"items": [{"id": "A", "name": "Widget"}, {"id": "B", "name": "Gadget"}]},
    items_cfg(ObjectFieldConfig(fields={"name": FieldConfig(key_metric=NormalizedMatch())})),
)
m2 = r2.field_scores["items"].metrics
assert m2["array_precision"] == 0.5 and m2["array_recall"] == 0.5 and m2["array_f1"] == 0.5

# ── Выровнялись по id, но контент неверный → hard 0, soft = accuracy ─────────
def wrong(metric):
    return evaluate(
        {"items": [{"id": "A", "name": "WRONG"}]},
        {"items": [{"id": "A", "name": "Widget"}]},
        EvalConfig(metrics=[metric], fields={"items": ArrayFieldConfig(
            strategy=ArrayStrategy.BY_KEY, key="id",
            item=ObjectFieldConfig(fields={"name": FieldConfig(key_metric=NormalizedMatch())}))}),
    ).field_scores["items"].metrics
assert wrong(ArrayPRF1())["array_f1"] == 0.0              # hard: element_score 0.5 < 1.0
assert wrong(ArrayPRF1(mode="soft"))["array_f1"] == 0.5  # soft: вклад = accuracy (id✓, name✗)
assert wrong(ArrayAccuracy())["array_accuracy"] == 0.5

# ── Root-level массив (BY_INDEX скаляров) ────────────────────────────────────
r3 = evaluate([1, 2, 3], [1, 2, 3], EvalConfig(metrics=[ArrayF1()]))
assert r3.metrics["array_f1"] == 1.0 and r3.field_scores["$"].node_type == "array"

# ── Вложенный массив (массив объектов с массивом-полем) ──────────────────────
r4 = evaluate(
    {"orders": [{"id": 1, "tags": ["a", "b"]}]},
    {"orders": [{"id": 1, "tags": ["a", "b"]}]},
    EvalConfig(metrics=[ArrayPRF1()], fields={"orders": ArrayFieldConfig(
        strategy=ArrayStrategy.BY_KEY, key="id",
        item=ObjectFieldConfig(fields={"tags": ArrayFieldConfig(strategy=ArrayStrategy.BY_INDEX)}))}),
)
assert r4.field_scores["orders"].metrics["array_f1"] == 1.0
assert "orders[0].tags[1]" in r4.field_scores

print("Stage 7 smoke OK")
```

## Запуск

```bash
python -c "$(sed -n '/^```python$/,/^```$/p' tests/smoke/stage7.md | sed '1d;$d')"
# → Stage 7 smoke OK
```
