# Stage 4 — Smoke test: матчеры как классы + реестр

**Цель этапа**: заменить `MatchMode` namespace и dataclass-матчеры на классы с
протоколом `Matcher` и авто-реестром через `__init_subclass__`.

## Изменения относительно плана

- **`NormalizedMatcher` → `RegexNormalizedMatcher`** с инъекцией regex через `__init__`.
  Дефолт — `pattern=r"\s+", repl=" "` («lowercase + strip + collapse spaces», как в доках).
  Кто хочет игнорировать пунктуацию — прокидывает свой regex. `name` остаётся `"NORMALIZED"`.
- **`NumericMatcher` — бинарный** (в пределах tolerance → 1.0, иначе 0.0). Соответствует
  v2 user-stories («в tolerance=0.01 → 1.0», US-2.9/US-3.5). Старая линейная формула
  `1 - dev/tol` давала на границе 0.99, что противоречило этим историям.
- **Нет `CustomMatcher(fn)`.** Кастомный матчер — это пользовательский класс, наследующий
  `MatcherBase` с `name` и `similarity()`; он автоматически регистрируется и передаётся
  как обычный `Matcher` (см. `DateMatcher` ниже).
- Smoke-пример из `implementation_stages_v1.md` для NUMERIC использовал `(100, 101)→0.0`,
  но 1/101 ≈ 0.99% < 1% → попадает в tolerance. Здесь взяты однозначные числа.

## Что проверяем

- Каждый матчер — класс с `name` и `similarity(actual, expected) -> float`.
- `RegexNormalizedMatcher` настраивается через regex.
- `NumericMatcher` — бинарный, relative/absolute, нечисловые → 0.0.
- `__init_subclass__`-реестр: `get_matcher_class(name)` разрешает имя в класс.
- `detect_matcher(key, value)` возвращает экземпляр матчера.
- Кастомный матчер — пользовательский подкласс `MatcherBase`, регистрируется сам.

## Скрипт (проверено — проходит)

```python
from structured_eval.matchers import (
    ExactMatcher, RegexNormalizedMatcher, NumericMatcher, TokenF1Matcher,
    JaccardMatcher, UrlMatcher, MatcherBase, get_matcher_class, detect_matcher,
)

# EXACT
assert ExactMatcher().similarity("a", "a") == 1.0
assert ExactMatcher().similarity("a", "b") == 0.0

# NUMERIC — бинарный в пределах tolerance
assert NumericMatcher(tolerance=0.01).similarity(100.0, 100.005) == 1.0  # 0.005% в полосе
assert NumericMatcher(tolerance=0.01).similarity(100.0, 102.0) == 0.0    # 2% вне полосы
assert NumericMatcher(tolerance=0).similarity(5, 5) == 1.0               # точное равенство
assert NumericMatcher(tolerance=0.01, mode="absolute").similarity(100, 100.5) == 0.0
assert NumericMatcher().similarity("x", 1) == 0.0                        # нечисловое

# NORMALIZED (regex-нормализация)
m = RegexNormalizedMatcher()                       # дефолт: схлопывает пробелы
assert m.similarity("Acme   Corp", "acme corp") == 1.0
assert m.similarity("Acme Corp.", "acme corp") == 0.0   # точка остаётся
mp = RegexNormalizedMatcher(pattern=r"[^\w\s]", repl="")  # убрать пунктуацию
assert mp.similarity("Acme Corp.", "acme corp") == 1.0

# TOKEN_F1 — непрерывный
assert TokenF1Matcher().similarity("the quick fox", "the quick fox") == 1.0
assert 0.0 < TokenF1Matcher().similarity("the quick fox", "the quick") < 1.0

# JACCARD / URL
assert JaccardMatcher().similarity("a b", "a b") == 1.0
assert UrlMatcher().similarity("https://A.com/path/", "https://a.com/path") == 1.0

# Кастомный матчер — пользовательский класс через MatcherBase
class DateMatcher(MatcherBase):
    name = "DATE"

    def similarity(self, actual, expected) -> float:
        from datetime import datetime
        fmt = "%Y-%m-%d"
        try:
            return 1.0 if datetime.strptime(str(actual), fmt) == datetime.strptime(str(expected), fmt) else 0.0
        except ValueError:
            return 0.0

assert DateMatcher().similarity("2024-01-15", "2024-01-15") == 1.0
assert DateMatcher().similarity("2024-01-15", "2024-01-16") == 0.0

# Реестр имён (включая авто-зарегистрированный кастомный)
assert get_matcher_class("EXACT") is ExactMatcher
assert get_matcher_class("NUMERIC") is NumericMatcher
assert get_matcher_class("NORMALIZED") is RegexNormalizedMatcher
assert get_matcher_class("DATE") is DateMatcher

# Autodetect → экземпляры
assert isinstance(detect_matcher("total", 100.0), NumericMatcher)
assert isinstance(detect_matcher("vendor_url", "http://x"), UrlMatcher)
assert isinstance(detect_matcher("status", "paid"), ExactMatcher)
assert isinstance(detect_matcher("description", "long text"), TokenF1Matcher)
assert isinstance(detect_matcher("name", "Acme"), TokenF1Matcher)

print("Stage 4 smoke OK")
```

## Запуск

```bash
python -c "$(sed -n '/^```python$/,/^```$/p' tests/smoke/stage4.md | sed '1d;$d')"
# → Stage 4 smoke OK
```
