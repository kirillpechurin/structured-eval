# Stage 9 — Smoke test: batch + consistency

**Цель этапа**: агрегация по нескольким документам.

- `evaluate(list[Sample])` → `BatchEvalReport` (mean-метрики, `perfect_response_rate`,
  `parse_error_rate`, `field_breakdown()`).
- `evaluate_consistency(runs=[...])` → `ConsistencyReport` (`field_variance`,
  `stable_fields`/`unstable_fields`, `score_variance`).

## Решения этапа

- **batch vs array-root**: `list[Sample]` — это батч; «голый» `list` как `actual` —
  это один документ с корнем-массивом (его оборачивают в `Sample`). Диспетчер —
  `_is_batch` (все элементы `Sample`).
- **`perfect_response_rate`** — доля семплов, которые распарсились И не имеют
  ни одного провалившегося поля (`failed_fields()` пуст). Не зависит от наличия
  `key_metric`.
- **mean-метрики** считаются только по успешно распарсенным семплам; `parse_error_rate`
  отдельно.
- **`field_breakdown()`** — per-path `mean/min/max/p95/fail_rate` (только узлы со score).
- **consistency**: `field_variance` — popul. дисперсия (`pvariance`) score поля между
  запусками; стабильно при `var <= variance_threshold` (дефолт 0.05).

## Скрипт (проверено — проходит)

```python
from structured_eval import (
    evaluate, evaluate_consistency, EvalConfig, FieldConfig, Sample,
    BatchEvalReport, ConsistencyReport,
)
from structured_eval.metrics import OverallScore, ExactMatch

cfg = EvalConfig(
    metrics=[OverallScore()],
    key_metric=OverallScore(),
    fields={
        "vendor": FieldConfig(key_metric=ExactMatch()),
        "total":  FieldConfig(key_metric=ExactMatch()),
    },
)

# ── Batch: 3 ok + 1 parse error ──────────────────────────────────────────────
samples = [
    Sample(actual={"vendor": "Acme", "total": 100}, expected={"vendor": "Acme", "total": 100}),  # perfect
    Sample(actual={"vendor": "Acme", "total": 999}, expected={"vendor": "Acme", "total": 100}),  # total wrong
    Sample(actual={"vendor": "X",    "total": 100}, expected={"vendor": "Acme", "total": 100}),  # vendor wrong
    Sample(actual="{ broken json",                  expected={"vendor": "Acme", "total": 100}),  # parse error
]
batch = evaluate(samples, cfg)
assert isinstance(batch, BatchEvalReport)
assert len(batch.per_sample) == 4
assert batch.parse_error_rate == 0.25
assert batch.perfect_response_rate == 0.25      # 1 of 4 fully correct
# mean overall_score over the 3 parsed: 1.0 + 0.5 + 0.5 = 2.0 / 3
assert abs(batch.metrics["overall_score"] - 2.0 / 3) < 1e-9
assert abs(batch.score - 2.0 / 3) < 1e-9 and batch.score_label == "overall_score"

bd = batch.field_breakdown()
assert bd["vendor"]["mean"] == 2 / 3 and bd["vendor"]["fail_rate"] == 1 / 3   # 1 of 3 wrong
assert bd["total"]["mean"] == 2 / 3 and bd["total"]["fail_rate"] == 1 / 3
assert bd["vendor"]["max"] == 1.0 and bd["vendor"]["min"] == 0.0

# ── Consistency: vendor stable, total jittery ────────────────────────────────
gt = {"vendor": "Acme", "total": 100}
runs = [
    Sample(actual={"vendor": "Acme", "total": 100}, expected=gt),
    Sample(actual={"vendor": "Acme", "total": 999}, expected=gt),
    Sample(actual={"vendor": "Acme", "total": 100}, expected=gt),
]
con = evaluate_consistency(runs=runs, config=cfg)
assert isinstance(con, ConsistencyReport)
assert con.field_variance["vendor"] == 0.0
assert con.field_variance["total"] > 0.0
assert "vendor" in con.stable_fields and "total" in con.unstable_fields
assert con.mean_score is not None

print("Stage 9 smoke OK")
```

## Запуск

```bash
python -c "$(sed -n '/^```python$/,/^```$/p' tests/smoke/stage9.md | sed '1d;$d')"
# → Stage 9 smoke OK
```
