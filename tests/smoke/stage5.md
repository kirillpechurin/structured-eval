# Stage 5 — Smoke test: метрики v3 (field + object)

**Цель этапа** (переосмыслен под technical_details_v3): сравнение поля — это
**field-метрика** (`ExactMatch`, `NormalizedMatch`, `Numeric`, `TokenF1`, `Fuzzy`,
`Presence`, `TypeMatch`), на поле их может быть несколько. Object-метрики
(`ObjectAccuracy`, `ObjectPrecision/Recall/F1/PRF1`, `ObjectValidity`) считают качество
как набора фактов; критерий совпадения — параметр метрики (`score_policy`/`threshold`)
с дефолтом из `ScalarNode.key_metric`, плюс `mode="soft"` без порога.

## Изменения относительно v2

- **`FieldPair`/`similarity` удалены.** `FieldMetric.compute(node)` делегирует в
  `score(actual, expected)` — чистый value-level примитив (переиспользуется выравниванием
  массива в Stage 7). Валидирующие метрики (`Presence`) переопределяют `compute`.
- **`FieldAccuracy` убрана** — её роль играет любая field-метрика напрямую.
- **`ScalarNode`** несёт `key_metric` + `threshold` (резолвится в match-фазе, Stage 6),
  а не `FieldPair`.
- **Object-метрики** принимают `score_policy` (dict поле→метрика/имя), `threshold`
  (dict или float) и `mode` (`hard`/`soft`); порядок разрешения вердикта —
  `score_policy` → `key_metric` поля → `ExactMatch@1.0`.
- Каждая метрика — отдельный модуль (field-метрики в `metrics/field/`).

## Скрипт (проверено — проходит)

```python
from structured_eval.core.config import EvalConfig
from structured_eval.core.context import EvalContext
from structured_eval.nodes.object_node import ObjectNode
from structured_eval.nodes.scalar import ScalarNode
from structured_eval.metrics import (
    ExactMatch, NormalizedMatch, Numeric, TokenF1, TypeMatch, Presence,
    FieldMetric, ObjectMetric,
    ObjectAccuracy, ObjectPrecision, ObjectRecall, ObjectF1, ObjectPRF1, ObjectValidity,
    get_metric_class,
)

def ctx(actual, expected):
    return EvalContext(actual=actual, expected=expected, source=None,
                       flat_actual={}, flat_expected={}, config=EvalConfig())

# ── Field-метрики: score(actual, expected) ──────────────────────────────────
assert ExactMatch().score("a", "a") == 1.0 and ExactMatch().score("a", "b") == 0.0
assert NormalizedMatch().score("Acme  Corp", "acme corp") == 1.0       # collapse+lower
assert NormalizedMatch().score("Acme Corp.", "Acme Corp") == 0.0       # пунктуация остаётся
assert NormalizedMatch(pattern=r"[^\w\s]", repl="").score("Acme Corp.", "Acme Corp") == 1.0
assert Numeric(tolerance=0.01).score(100, 100.005) == 1.0
assert Numeric(tolerance=0.01).score(100, 102) == 0.0
assert abs(TokenF1().score("red car", "a red sports car") - 2 / 3) < 1e-9
assert TypeMatch().score(100, 100) == 1.0
assert TypeMatch().score("100", 100) == 0.0     # строка вместо числа
assert TypeMatch().score(True, 1) == 0.0        # bool ≠ number

# compute(node) делегирует в score; Presence переопределяет compute
c0 = ctx({"vendor": "Acme"}, {"vendor": "Acme"})
assert ExactMatch().compute(ScalarNode("vendor", c0)) == 1.0
assert Presence().compute(ScalarNode("vendor", c0)) == 1.0
assert Presence().compute(ScalarNode("ghost", c0)) == 0.0   # отсутствует → None → 0.0

# ── Типы и реестр ────────────────────────────────────────────────────────────
assert isinstance(ExactMatch(), FieldMetric)
assert isinstance(ObjectF1(), ObjectMetric)
assert get_metric_class("exact_match") is ExactMatch
assert get_metric_class("token_f1") is TokenF1
assert get_metric_class("object_prf1") is ObjectPRF1

# ── Object: 2 верных, 1 missing, 1 spurious → P=R=F1=2/3 (дефолт exact) ──────
c1 = ctx({"vendor": "Acme", "total": 100, "note": "x"},
         {"vendor": "Acme", "total": 100, "status": "paid"})
obj = ObjectNode("$", c1,
                 matched=[ScalarNode("vendor", c1), ScalarNode("total", c1)],
                 missing=["status"], spurious=["note"])
assert abs(ObjectPrecision().compute(obj) - 2 / 3) < 1e-9
assert abs(ObjectRecall().compute(obj)   - 2 / 3) < 1e-9
assert abs(ObjectF1().compute(obj)       - 2 / 3) < 1e-9
assert abs(ObjectAccuracy().compute(obj) - 2 / 3) < 1e-9
prf = ObjectPRF1().compute(obj)
assert set(prf) == {"object_precision", "object_recall", "object_f1"}
assert abs(prf["object_f1"] - 2 / 3) < 1e-9
assert ObjectValidity().compute(obj) == 1.0   # оба присутствующих — верного типа

# ── Сценарий 1: критерий совпадения на уровне поля (key_metric) ──────────────
cb = ctx({"a": "Hello World", "b": "Acme Corp."},
         {"a": "hello world", "b": "Acme Corp"})
# дефолт exact: оба отличаются → F1 0
obj_default = ObjectNode("$", cb, matched=[ScalarNode("a", cb), ScalarNode("b", cb)])
assert ObjectF1().compute(obj_default) == 0.0
# key_metric на полях поднимает оба до совпадения → F1 1.0
obj_keyed = ObjectNode("$", cb, matched=[
    ScalarNode("a", cb, key_metric=NormalizedMatch()),
    ScalarNode("b", cb, key_metric=NormalizedMatch(pattern=r"[^\w\s]", repl="")),
])
assert ObjectF1().compute(obj_keyed) == 1.0

# ── Сценарий 2: score_policy на метрике переопределяет поля ───────────────────
f1_override = ObjectF1(
    score_policy={"a": "normalized_match", "b": "token_f1"},
    threshold={"a": 1.0, "b": 1.0},
)
assert f1_override.compute(obj_default) == 1.0   # те же ноды (key_metric=None), критерий с метрики

# ── Soft-режим: без порога, дробный вклад score ──────────────────────────────
cc = ctx({"x": "red car"}, {"x": "a red sports car"})
obj_soft = ObjectNode("$", cc, matched=[ScalarNode("x", cc, key_metric=TokenF1())])
assert ObjectF1().compute(obj_soft) == 0.0                          # hard: 2/3 < порог 1.0
assert abs(ObjectF1(mode="soft").compute(obj_soft) - 2 / 3) < 1e-9  # soft: вклад = score

print("Stage 5 (v3) smoke OK")
```

## Запуск

```bash
python -c "$(sed -n '/^```python$/,/^```$/p' tests/smoke/stage5.md | sed '1d;$d')"
# → Stage 5 (v3) smoke OK
```
