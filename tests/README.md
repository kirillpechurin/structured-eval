# Test suite — architecture & conventions

The canonical style for tests in this project. When a test and this document
disagree, the test is wrong — fix the test.

## 1. Layout mirrors the source tree

The test tree mirrors `structured_eval/` one-to-one, layer by layer:

```
structured_eval/metrics/exact.py      → tests/unit/metrics/test_exact.py
structured_eval/alignment/by_key.py   → tests/unit/test_alignment_by_key.py
structured_eval/models/nodes/scalar.py → tests/unit/models/test_nodes_scalar.py
```

**One file = one unit under test.** Never mix independent units in one file.
"Unit" is the *cohesive* unit, not the Python symbol:

- Each independent scalar metric is its own file (`test_exact.py`,
  `test_numeric.py`, `test_fuzzy.py`, …).
- A metric **family** that shares one mechanism is one file: the object
  precision/recall/F1/accuracy metrics all read the same confusion matrix →
  `test_object_metrics.py`. Likewise `test_array_metrics.py`. They are three
  projections of one unit, not independent units.
- A metric with a helper package mirrors the package: `schema_validity/` →
  `test_schema_validity.py`.

The layers and what they exercise:

| Dir                  | Layer            | What it tests                                  |
|----------------------|------------------|------------------------------------------------|
| `unit/`              | pure functions   | one module, no engine; `make_context`/`build_tree` |
| `engine/`            | integration      | the full `Evaluator` pipeline end-to-end       |
| `api/`               | public surface   | `evaluate*` call shapes / modes                |
| `golden/`            | system regression| pinned numbers on dataset fixtures             |
| `properties/`        | system invariants| generative property / metamorphic tests        |
| `integration/`       | adapters         | deepeval / langsmith (skip without extras)     |

## 2. Flat functions + `parametrize` — no test classes

Tests are **module-level functions**, never grouped in `class Test…`. Grouping
is the file's job (rule 1), so classes add a redundant layer. Shared setup is a
fixture; shared data is a builder.

**Table-driven by default.** If cases differ only in data, they are one
`@pytest.mark.parametrize` with explicit `ids`, not N near-duplicate functions:

```python
@pytest.mark.parametrize(
    ("actual", "expected", "score"),
    [
        ("paid", "paid", 1.0),
        ("paid", "draft", 0.0),
        ("100", 100, 0.0),     # type-sensitive
        (None, None, 1.0),
    ],
    ids=["equal", "differ", "type-sensitive", "null-eq-null"],
)
def test_score(actual, expected, score):
    assert ExactMatch().score(actual, expected) == score
```

A case that needs its own narrative (distinct arrange/act, a raised error, a
multi-step interaction) stays a standalone function — don't force it into a
table.

## 3. Naming & shape

- `test_<scenario>` — the scenario, not the unit (the file/symbol already names
  the unit): `test_currency_symbols_stripped`, not `test_numeric_currency`.
- **Arrange / Act / Assert**, in that order, with one logical assertion per test
  (tightly-coupled checks like precision+recall together are fine).
- Comments explain the **arithmetic / why** (`tp=1, predicted=2 → 0.5`), never
  the *what*.

## 4. Shared infrastructure (`conftest.py`)

Use these instead of re-deriving boilerplate:

- **Engine helpers** — `evaluate_one` (≙ `run`), `context_factory`, `tree_factory`.
- **Builders** — `make_invoice(**overrides)` / `invoice_builder`; build domain
  data showing only the field under test.
- **Semantic assertions** — `assert_metric(report, "object_f1", 0.5)` and
  `assert_field(report, "total", 0.0)`; prefer them over reaching into
  `report.metrics[...].representative()` by hand.

## 5. Contracts & properties

- `unit/metrics/test_metric_contracts.py` runs **every registered metric**
  through baseline invariants (bounded, total, reflexive). A new metric inherits
  these for free — do not re-assert them per metric.
- `properties/` holds generative invariants (seeded, `parametrize` over `SEEDS`).
  Every failure is reproducible from the seed printed in the test id. No
  third-party PBT library — generators live in `properties/conftest.py`.

## 6. Markers

`unit` / `engine` / `integration` / `golden` / `property`. Set once per file:
`pytestmark = pytest.mark.<layer>`.

## 7. Running

```bash
uv run pytest                       # full suite
uv run pytest -m unit               # one layer
uv run pytest tests/unit/metrics    # one area
uv run pytest --cov=structured_eval # coverage (fail_under in pyproject)
```
