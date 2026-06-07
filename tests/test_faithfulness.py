import pytest

from structured_eval import EvalConfig, FieldConfig, evaluate
from structured_eval.faithfulness.substring import compute_faithfulness


SOURCE = "Invoice from Acme Corp, total amount 100.0 USD, status paid, date 2024-01-15"


# ── compute_faithfulness directly ─────────────────────────────────────────────


class TestComputeFaithfulness:
    def test_all_found(self):
        actual = {"vendor": "Acme Corp", "total": "100.0"}
        score, hallucinated = compute_faithfulness(actual, SOURCE, EvalConfig())
        assert score == 1.0
        assert hallucinated == []

    def test_none_found(self):
        actual = {"vendor": "Fake Corp", "currency": "BTC"}
        score, hallucinated = compute_faithfulness(actual, SOURCE, EvalConfig())
        assert score == 0.0
        assert set(hallucinated) == {"vendor", "currency"}

    def test_partial(self):
        actual = {"vendor": "Acme Corp", "currency": "BTC"}
        score, hallucinated = compute_faithfulness(actual, SOURCE, EvalConfig())
        assert score == pytest.approx(0.5)
        assert "currency" in hallucinated
        assert "vendor" not in hallucinated

    def test_case_insensitive(self):
        actual = {"status": "PAID"}
        score, _ = compute_faithfulness(actual, SOURCE, EvalConfig())
        assert score == 1.0

    def test_derived_excluded(self):
        cfg = EvalConfig(fields={"total": FieldConfig(derived=True)})
        actual = {"vendor": "Acme Corp", "total": "999.0"}  # total not in source this way
        score, hallucinated = compute_faithfulness(actual, SOURCE, cfg)
        # total is derived → only vendor is checked
        assert score == 1.0
        assert "total" not in hallucinated

    def test_empty_actual_returns_perfect(self):
        score, hallucinated = compute_faithfulness({}, SOURCE, EvalConfig())
        assert score == 1.0
        assert hallucinated == []

    def test_none_values_skipped(self):
        actual = {"vendor": "Acme Corp", "missing_field": None}
        score, hallucinated = compute_faithfulness(actual, SOURCE, EvalConfig())
        assert "missing_field" not in hallucinated

    def test_nested_fields(self):
        actual = {"vendor": {"name": "Acme Corp", "country": "USD"}}
        source = "Vendor: Acme Corp, currency USD"
        score, hallucinated = compute_faithfulness(actual, source, EvalConfig())
        assert score == 1.0

    def test_nested_fields_partial(self):
        actual = {"vendor": {"name": "Acme Corp", "country": "UNKNOWN"}}
        source = "Vendor: Acme Corp"
        score, hallucinated = compute_faithfulness(actual, source, EvalConfig())
        assert score == pytest.approx(0.5)
        assert "vendor.country" in hallucinated

    def test_list_of_primitives(self):
        actual = {"tags": ["Acme", "Corp"]}
        source = "Acme Corp invoice"
        score, hallucinated = compute_faithfulness(actual, source, EvalConfig())
        assert score == 1.0

    def test_list_partial(self):
        actual = {"tags": ["Acme", "Unknown"]}
        source = "Acme Corp invoice"
        score, hallucinated = compute_faithfulness(actual, source, EvalConfig())
        assert score == pytest.approx(0.5)
        assert "tags[1]" in hallucinated


# ── via evaluate() ────────────────────────────────────────────────────────────


class TestEvaluateFaithfulness:
    def test_faithfulness_with_expected(self):
        actual = {"vendor": "Acme Corp", "total": 100.0}
        expected = {"vendor": "Acme Corp", "total": 100.0}
        report = evaluate(actual, expected, source=SOURCE, detailed=True)
        assert report.faithfulness_score == 1.0
        assert report.hallucinated_fields == []
        # f1 is also computed since expected was provided
        assert report.f1 is not None

    def test_faithfulness_without_expected(self):
        actual = {"vendor": "Acme Corp"}
        report = evaluate(actual, source=SOURCE, config=EvalConfig(), detailed=True)
        assert report.faithfulness_score == 1.0
        assert report.f1 is None

    def test_hallucinated_fields_populated(self):
        actual = {"vendor": "Ghost Corp", "status": "paid"}
        expected = {"vendor": "Acme Corp", "status": "paid"}
        report = evaluate(actual, expected, source=SOURCE, detailed=True)
        assert "vendor" in report.hallucinated_fields
        assert "status" not in report.hallucinated_fields

    def test_faithfulness_none_when_no_source(self):
        actual = {"vendor": "Acme Corp"}
        expected = {"vendor": "Acme Corp"}
        report = evaluate(actual, expected, detailed=True)
        assert report.faithfulness_score is None
        assert report.hallucinated_fields == []

    def test_derived_excluded_via_evaluate(self):
        actual = {"vendor": "Acme Corp", "total": 999.0}
        expected = {"vendor": "Acme Corp", "total": 999.0}
        report = evaluate(
            actual,
            expected,
            source=SOURCE,
            config=EvalConfig(fields={"total": FieldConfig(derived=True)}),
            detailed=True,
        )
        # total derived → excluded; only vendor checked → score 1.0
        assert report.faithfulness_score == 1.0
        assert "total" not in report.hallucinated_fields
