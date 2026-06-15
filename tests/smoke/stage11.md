# Stage 11 — Smoke test: report formats (console + JSON/diff)

**Цель этапа**: вывод отчётов. MVP: console-рендер (`print_summary`) для single/batch/
consistency + JSON round-trip (`to_json`/`from_json`/`to_dict`/`from_dict`) + `diff_from`.
JUnit/HTML и `assert_*` — на потом.

## Решения этапа

- **Console — чистый stdlib** (Unicode box-drawing), без жёсткой зависимости: `print_summary`
  работает из коробки. Rich — опциональный слой для цвета на будущее (extra `report`).
- **Таблица полей — все поля** (по решению дизайна): скалярные листья со score → `Field |
  Metric | Score | Bar | ✓/✗`; имя метрики восстанавливается reverse-lookup-ом значения в
  `fs.metrics`. Агрегаты объектов/массивов — в секции `Structure`.
- **None-score** (нет ground truth) рендерится как `—`, не FAIL.
- **JSON round-trip**: `ArrayMatchResult.matched` — кортежи, при загрузке восстанавливаются
  (для равенства); `ArrayStrategy` (StrEnum) ↔ строка.
- **`diff_from`** считает `self - other`: document-метрики, общие для обоих отчётов (или
  подмножество по `metrics=[...]`), и per-field дельты (включая `score`).

## Скрипт (проверено — проходит)

```python
import tempfile, os
from structured_eval import (
    evaluate, EvalConfig, FieldConfig, Sample, BatchEvalReport,
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

# ── console render: single ───────────────────────────────────────────────────
from structured_eval.report import render
r = evaluate({"vendor": "X", "total": 100}, {"vendor": "Acme", "total": 100}, cfg)
text = render(r)
assert "OVERALL" in text and "0.50" in text
assert "vendor" in text and "✗" in text          # vendor failed
assert "Failures" in text and "'X'" in text and "'Acme'" in text
r.print_summary()                                 # must not raise

# ── JSON round-trip ──────────────────────────────────────────────────────────
d = r.to_dict()
from structured_eval import EvalReport
assert EvalReport.from_dict(d) == r               # dataclass eq

with tempfile.TemporaryDirectory() as tmp:
    p = os.path.join(tmp, "report.json")
    r.to_json(p)
    loaded = EvalReport.from_json(p)
assert loaded == r
assert loaded.failed_fields()[0].path == "vendor"  # queries still work

# ── diff_from ────────────────────────────────────────────────────────────────
v1 = evaluate({"vendor": "X",    "total": 999}, {"vendor": "Acme", "total": 100}, cfg)  # both wrong → 0.0
v2 = evaluate({"vendor": "Acme", "total": 100}, {"vendor": "Acme", "total": 100}, cfg)  # both right → 1.0
diff = v2.diff_from(v1)
assert diff.deltas["overall_score"] == 1.0                    # improved
assert diff.field_deltas["vendor"]["score"] == 1.0
assert diff.field_deltas["total"]["exact_match"] == 1.0
# regression direction: v1 vs v2 is negative
assert v1.diff_from(v2).deltas["overall_score"] == -1.0
# metric subset selection
assert list(v2.diff_from(v1, metrics=["overall_score"]).deltas) == ["overall_score"]

# ── batch / consistency render ───────────────────────────────────────────────
batch = evaluate([
    Sample(actual={"vendor": "Acme", "total": 100}, expected={"vendor": "Acme", "total": 100}),
    Sample(actual={"vendor": "X",    "total": 100}, expected={"vendor": "Acme", "total": 100}),
], config=cfg)
btext = render(batch)
assert "BATCH" in btext and "perfect_response_rate" in btext and "fail_rate" in btext
batch.print_summary()

from structured_eval import evaluate_consistency
con = evaluate_consistency(runs=[
    Sample(actual={"vendor": "Acme", "total": 100}, expected={"vendor": "Acme", "total": 100}),
    Sample(actual={"vendor": "Acme", "total": 999}, expected={"vendor": "Acme", "total": 100}),
], config=cfg)
ctext = render(con)
assert "CONSISTENCY" in ctext and "variance" in ctext
con.print_summary()

print("Stage 11 smoke OK")
```

## Запуск

```bash
python -c "$(sed -n '/^```python$/,/^```$/p' tests/smoke/stage11.md | sed '1d;$d')"
# → Stage 11 smoke OK
```
