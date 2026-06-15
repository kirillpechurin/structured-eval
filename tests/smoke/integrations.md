# Integrations — Smoke test: deepeval / langsmith adapters

**Цель**: structured-eval как метрика/evaluator внутри хост-раннеров. Пользователь
не переезжает — подключает field-level оценку в свой deepeval/langsmith pipeline.

## Решения

- **Ленивый импорт хост-либ**: `import structured_eval` и даже `import
  structured_eval.integrations.deepeval` безопасны без установленных deepeval/langsmith.
  Класс `StructuredMetric` строится лениво через module `__getattr__` (тянет
  `deepeval.metrics.BaseMetric` только при обращении).
- **Только overall score + reason** (по решению дизайна): адаптеры отдают `report.score`
  и текст с провалившимися полями (`failed_fields`), без per-field/per-metric разбивки.
- **Общий слой** `_adapter.verdict(report, threshold) -> (score, success, reason)` —
  фреймворк-агностичный, тестируется без хост-либ. `success` требует распарсенного
  документа и `score >= threshold`; `score is None` → `success=False`.
- **langsmith**: фабрика `structured_evaluator(config, ...)` → callable `(run, example)`
  → `{"key","score","comment"}`. По умолчанию actual/expected берутся из `.outputs`;
  хуки `extract_actual`/`extract_expected` — для вложенных полей/иной формы.
- extras: `structured-eval[deepeval]`, `[langsmith]` (в `all` не включены — тяжёлые
  фреймворк-зависимости).

## Скрипт (проверено — проходит)

```python
from structured_eval import evaluate, EvalConfig, FieldConfig
from structured_eval.metrics import OverallScore, ExactMatch
from structured_eval.integrations import verdict, reason_text

cfg = EvalConfig(
    metrics=[OverallScore()],
    key_metric=OverallScore(),
    fields={
        "vendor": FieldConfig(key_metric=ExactMatch()),
        "total":  FieldConfig(key_metric=ExactMatch()),
    },
)

# ── agnostic adapter: verdict + reason ───────────────────────────────────────
good = evaluate({"vendor": "Acme", "total": 100}, {"vendor": "Acme", "total": 100}, cfg)
score, success, reason = verdict(good, 0.85)
assert score == 1.0 and success is True and reason == "all fields passed"

bad = evaluate({"vendor": "X", "total": 100}, {"vendor": "Acme", "total": 100}, cfg)
score, success, reason = verdict(bad, 0.85)
assert score == 0.5 and success is False
assert "vendor" in reason and "'X'" in reason and "'Acme'" in reason

perr = evaluate("{ broken", {"vendor": "Acme"}, cfg)
score, success, reason = verdict(perr, 0.0)
assert score is None and success is False and reason.startswith("parse error")

# ── langsmith evaluator (duck-typed run/example with .outputs) ───────────────
from structured_eval.integrations.langsmith import structured_evaluator

class Box:
    def __init__(self, outputs): self.outputs = outputs

ev = structured_evaluator(config=cfg, threshold=0.85, key="invoice_eval")
res = ev(Box({"vendor": "X", "total": 100}), Box({"vendor": "Acme", "total": 100}))
assert res == {"key": "invoice_eval", "score": 0.5, "comment": reason} or (
    res["key"] == "invoice_eval" and res["score"] == 0.5 and "vendor" in res["comment"]
)
assert ev.__name__ == "invoice_eval"

# extraction hooks: pull a nested field
ev2 = structured_evaluator(
    config=cfg,
    extract_actual=lambda run: run.outputs["prediction"],
    extract_expected=lambda ex: ex.outputs["gold"],
)
res2 = ev2(
    Box({"prediction": {"vendor": "Acme", "total": 100}}),
    Box({"gold": {"vendor": "Acme", "total": 100}}),
)
assert res2["score"] == 1.0 and res2["comment"] == "all fields passed"

# ── deepeval module imports safely without the library installed ─────────────
import importlib
mod = importlib.import_module("structured_eval.integrations.deepeval")
try:
    mod.StructuredMetric            # triggers lazy build → needs deepeval
    deepeval_present = True
except (ModuleNotFoundError, ImportError):
    deepeval_present = False        # expected here: deepeval not installed
print(f"deepeval installed: {deepeval_present}")

print("Integrations smoke OK")
```

## Запуск

```bash
python -c "$(sed -n '/^```python$/,/^```$/p' tests/smoke/integrations.md | sed '1d;$d')"
# → Integrations smoke OK
```
