import pytest

from structured_eval.core.config import MatchMode
from structured_eval.metrics.matchers import detect_matcher, match


# ── JACCARD ───────────────────────────────────────────────────────────────────


class TestJaccard:
    def test_identical(self):
        assert match(MatchMode.JACCARD, "hello world", "hello world") == 1.0

    def test_partial_overlap(self):
        # actual = {the, quick, brown, fox}, expected = {the, quick, fox}
        # intersection = 3, union = 4 → 3/4
        assert match(MatchMode.JACCARD, "the quick brown fox", "the quick fox") == pytest.approx(3 / 4)

    def test_no_overlap(self):
        assert match(MatchMode.JACCARD, "apple orange", "banana grape") == 0.0

    def test_both_empty(self):
        assert match(MatchMode.JACCARD, "", "") == 1.0

    def test_actual_empty(self):
        assert match(MatchMode.JACCARD, "", "hello") == 0.0

    def test_expected_empty(self):
        assert match(MatchMode.JACCARD, "hello", "") == 0.0

    def test_case_insensitive(self):
        assert match(MatchMode.JACCARD, "Hello World", "hello world") == 1.0

    def test_subset(self):
        # actual ⊂ expected: {a} vs {a, b, c} → 1/3
        assert match(MatchMode.JACCARD, "a", "a b c") == pytest.approx(1 / 3)

    def test_symmetric(self):
        # Jaccard is symmetric
        a, b = "quick fox", "quick brown fox"
        assert match(MatchMode.JACCARD, a, b) == pytest.approx(match(MatchMode.JACCARD, b, a))

    def test_str_representation(self):
        assert str(MatchMode.JACCARD) == "jaccard"

    def test_singleton(self):
        assert MatchMode.JACCARD is MatchMode.JACCARD


# ── URL ───────────────────────────────────────────────────────────────────────


class TestUrl:
    def test_identical(self):
        assert match(MatchMode.URL, "https://example.com", "https://example.com") == 1.0

    def test_trailing_slash_normalized(self):
        assert match(MatchMode.URL, "https://example.com/", "https://example.com") == 1.0

    def test_trailing_slash_on_path(self):
        assert match(MatchMode.URL, "https://example.com/path/", "https://example.com/path") == 1.0

    def test_query_param_order(self):
        assert match(MatchMode.URL, "https://example.com?b=2&a=1", "https://example.com?a=1&b=2") == 1.0

    def test_different_hosts(self):
        assert match(MatchMode.URL, "https://example.com", "https://other.com") == 0.0

    def test_different_paths(self):
        assert match(MatchMode.URL, "https://example.com/a", "https://example.com/b") == 0.0

    def test_different_query(self):
        assert match(MatchMode.URL, "https://example.com?a=1", "https://example.com?a=2") == 0.0

    def test_path_vs_no_path(self):
        assert match(MatchMode.URL, "https://example.com/path", "https://example.com") == 0.0

    def test_str_representation(self):
        assert str(MatchMode.URL) == "url"

    def test_singleton(self):
        assert MatchMode.URL is MatchMode.URL


# ── detect_matcher updates ────────────────────────────────────────────────────


class TestDetectMatcherUpdates:
    def test_url_suffix(self):
        assert str(detect_matcher("website_url", "https://example.com")) == "url"

    def test_link_suffix(self):
        assert str(detect_matcher("profile_link", "https://example.com")) == "url"

    def test_href_field(self):
        assert str(detect_matcher("href", "https://example.com")) == "url"

    def test_uri_field(self):
        assert str(detect_matcher("uri", "https://example.com")) == "url"

    def test_description_suffix_not_exact(self):
        # Regression: country_description used to → EXACT, now → TOKEN_F1
        assert str(detect_matcher("country_description", "some text")) == "token_f1"

    def test_note_suffix(self):
        assert str(detect_matcher("item_note", "some note text")) == "token_f1"

    def test_text_suffix(self):
        assert str(detect_matcher("body_text", "some text")) == "token_f1"

    def test_plain_country_still_exact(self):
        # Plain "country" without text suffix → still EXACT
        assert str(detect_matcher("country", "US")) == "exact"

    def test_status_still_exact(self):
        assert str(detect_matcher("status", "paid")) == "exact"

    def test_url_takes_priority_over_exact(self):
        # "status_url" → URL wins over "status" EXACT
        assert str(detect_matcher("status_url", "https://example.com")) == "url"
