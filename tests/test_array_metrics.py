import pytest

from structured_eval.metrics.array_metrics import (
    array_element_precision,
    array_element_recall,
    array_exact_match,
    array_set_match,
)


class TestArrayExactMatch:
    def test_identical(self):
        assert array_exact_match([1, 2, 3], [1, 2, 3]) == 1.0

    def test_different_order(self):
        assert array_exact_match([1, 3, 2], [1, 2, 3]) == 0.0

    def test_different_values(self):
        assert array_exact_match([1, 2, 4], [1, 2, 3]) == 0.0

    def test_different_lengths(self):
        assert array_exact_match([1, 2], [1, 2, 3]) == 0.0

    def test_both_empty(self):
        assert array_exact_match([], []) == 1.0

    def test_strings(self):
        assert array_exact_match(["a", "b"], ["a", "b"]) == 1.0
        assert array_exact_match(["a", "b"], ["b", "a"]) == 0.0


class TestArrayElementRecall:
    def test_full_recall(self):
        assert array_element_recall(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_partial_recall(self):
        assert array_element_recall(["a", "b"], ["a", "b", "c"]) == pytest.approx(2 / 3)

    def test_zero_recall(self):
        assert array_element_recall(["x", "y"], ["a", "b"]) == 0.0

    def test_empty_expected(self):
        assert array_element_recall(["a"], []) == 1.0

    def test_empty_actual(self):
        assert array_element_recall([], ["a", "b"]) == 0.0

    def test_duplicates_counted(self):
        # actual has one "a", expected has two "a" → recall = 1/2
        assert array_element_recall(["a"], ["a", "a"]) == pytest.approx(0.5)

    def test_extra_in_actual_ignored(self):
        # actual has more elements than expected — recall stays 1.0
        assert array_element_recall(["a", "b", "extra"], ["a", "b"]) == 1.0


class TestArrayElementPrecision:
    def test_full_precision(self):
        assert array_element_precision(["a", "b"], ["a", "b", "c"]) == 1.0

    def test_partial_precision(self):
        # actual=["a","b","x"], expected=["a","b"] → 2 of 3 actual elements in expected
        assert array_element_precision(["a", "b", "x"], ["a", "b"]) == pytest.approx(2 / 3)

    def test_zero_precision(self):
        assert array_element_precision(["x", "y"], ["a", "b"]) == 0.0

    def test_empty_actual(self):
        assert array_element_precision([], ["a"]) == 1.0

    def test_empty_expected(self):
        assert array_element_precision(["a"], []) == 0.0

    def test_duplicates_counted(self):
        # actual has two "a", expected has one "a" → precision = 1/2
        assert array_element_precision(["a", "a"], ["a"]) == pytest.approx(0.5)


class TestArraySetMatch:
    def test_both_empty(self):
        assert array_set_match([], []) == 1.0

    def test_perfect_match(self):
        assert array_set_match(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_order_insensitive(self):
        assert array_set_match(["c", "a", "b"], ["a", "b", "c"]) == 1.0

    def test_partial_overlap(self):
        # actual=["a","b","x"], expected=["a","b","c"]
        # recall = 2/3, precision = 2/3, F1 = 2/3
        assert array_set_match(["a", "b", "x"], ["a", "b", "c"]) == pytest.approx(2 / 3)

    def test_no_overlap(self):
        assert array_set_match(["x", "y"], ["a", "b"]) == 0.0

    def test_subset_actual(self):
        # actual=["a"], expected=["a","b"] → recall=0.5, precision=1.0, F1=2/3
        assert array_set_match(["a"], ["a", "b"]) == pytest.approx(2 / 3)

    def test_duplicate_handling(self):
        # actual=[1,1,2], expected=[1,2,2]
        # intersection multiset = [1,2] → size 2
        # recall = 2/3, precision = 2/3, F1 = 2/3
        assert array_set_match([1, 1, 2], [1, 2, 2]) == pytest.approx(2 / 3)

    def test_integers(self):
        assert array_set_match([1, 2, 3], [1, 2, 3]) == 1.0
        assert array_set_match([1, 2], [1, 2, 3]) == pytest.approx(4 / 5)
