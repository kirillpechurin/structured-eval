# Stage 3 — Smoke test: EvalNode tree + Sample + EvalContext

**Цель этапа**: новые data structures (`nodes/`, `core/sample.py`, `core/context.py`,
переписанные `core/config.py` и `core/result.py`). Логики движка ещё нет — только структуры.

## Что проверяем

- `EvalNode` хранит `path` + ссылку на `EvalContext`; `actual`/`expected` — ленивый доступ через `_navigate`.
- `_navigate` ходит по dot-and-bracket путям; на отсутствующем пути возвращает `MISSING`.
- `ScalarNode` / `ObjectNode` / `ArrayNode` конструируются с корректными дефолтами.
- `ArrayMatchResult` вычисляет precision/recall/f1 из matched/missed/spurious.
- `Sample` разграничивает root-array (один документ) и batch (`list[Sample]`).
- `EvalReport.failed_fields()` учитывает field-level и переданный threshold.

## Скрипт (проверено — проходит)

```python
from structured_eval.core.config import ArrayStrategy, EvalConfig
from structured_eval.core.context import EvalContext
from structured_eval.core.result import EvalReport, FieldScore
from structured_eval.core.sample import Sample
from structured_eval.nodes.array_node import ArrayMatchResult, ArrayNode
from structured_eval.nodes.base import MISSING, EvalNode, _navigate
from structured_eval.nodes.object_node import ObjectNode
from structured_eval.nodes.scalar import FieldPair, ScalarNode
from structured_eval.utils.flatten import flatten

actual = {"vendor": "Acme", "total": 100.0}
expected = {"vendor": "Acme Corp", "total": 100.0}

ctx = EvalContext(
    actual=actual,
    expected=expected,
    source=None,
    flat_actual=flatten(actual),
    flat_expected=flatten(expected),
    config=EvalConfig(),
)

# 1. EvalNode — ленивый доступ
node = EvalNode(path="vendor", context=ctx)
assert node.actual == "Acme"
assert node.expected == "Acme Corp"
assert EvalNode(path="missing_field", context=ctx).actual is None

# 2. _navigate
assert _navigate(actual, "$") == actual
assert _navigate(actual, "vendor") == "Acme"
assert _navigate({"a": {"b": 1}}, "a.b") == 1
assert _navigate({"items": [1, 2]}, "items[0]") == 1
assert _navigate({"items": [{"name": "x"}]}, "items[0].name") == "x"
assert _navigate({"a": 1}, "b") is MISSING
assert _navigate({"items": [1]}, "items[5]") is MISSING
assert _navigate({"a": 1}, "a.b") is MISSING

# 3. expected=None → expected читается как None
ctx_no_exp = EvalContext(
    actual=actual, expected=None, source=None,
    flat_actual=flatten(actual), flat_expected={}, config=EvalConfig(),
)
assert EvalNode(path="vendor", context=ctx_no_exp).expected is None

# 4. Узлы
pair = FieldPair(actual="Acme", expected="Acme", matcher=None, similarity=1.0)
sn = ScalarNode(path="vendor", context=ctx, pair=pair)
assert sn.pair.similarity == 1.0 and sn.metric_results == {}

on = ObjectNode(path="$", context=ctx)
assert on.matched == [] and on.missing == [] and on.spurious == [] and on.children == {}

an = ArrayNode(path="items", context=ctx)
assert an.match_result is None and an.items == []

# 5. ArrayMatchResult — derived metrics
m = ArrayMatchResult(
    strategy=ArrayStrategy.BY_INDEX,
    matched=[(0, 0), (1, 1)], missed=[2], spurious=[],
)
assert m.precision == 1.0
assert m.recall == 2 / 3
assert abs(m.f1 - (2 * 1.0 * (2 / 3)) / (1.0 + 2 / 3)) < 1e-9

# 6. Sample — root-array vs batch
single = Sample(actual=[{"id": "A"}, {"id": "B"}])
assert isinstance(single.actual, list)
batch = [Sample(actual={"id": "A"}), Sample(actual={"id": "B"})]
assert all(isinstance(s, Sample) for s in batch)

# 7. EvalReport.failed_fields
report = EvalReport(field_scores={
    "a": FieldScore(path="a", node_type="scalar", score=1.0),
    "b": FieldScore(path="b", node_type="scalar", score=0.5),
    "c": FieldScore(path="c", node_type="scalar", score=0.9, threshold=0.8),
    "d": FieldScore(path="d", node_type="scalar", score=None),
})
assert {fs.path for fs in report.failed_fields()} == {"b"}
assert {fs.path for fs in report.failed_fields(threshold=0.95)} == {"b", "c"}

print("Stage 3 smoke OK")
```

## Запуск

```bash
python -c "$(sed -n '/^```python$/,/^```$/p' tests/smoke/stage3.md | sed '1d;$d')"
# → Stage 3 smoke OK
```
