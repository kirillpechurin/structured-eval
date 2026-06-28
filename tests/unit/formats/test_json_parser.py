"""JsonParser / JsonlParser (one module) + the shared ParseError contract."""

import pytest

from structured_eval.formats.base import ParseError
from structured_eval.formats.json_parser import JsonlParser, JsonParser

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("text", "expected"),
    [('{"a": 1}', {"a": 1}), ("[1, 2]", [1, 2]), ("42", 42)],
    ids=["object", "array", "scalar"],
)
def test_json_parses(text, expected):
    assert JsonParser().parse(text) == expected


def test_json_invalid_raises_parse_error():
    with pytest.raises(ParseError):
        JsonParser().parse("{not json}")


def test_jsonl_lines():
    assert list(JsonlParser().parse('{"a": 1}\n{"a": 2}')) == [{"a": 1}, {"a": 2}]


def test_jsonl_blank_lines_skipped():
    assert list(JsonlParser().parse('{"a": 1}\n\n{"a": 2}\n')) == [{"a": 1}, {"a": 2}]


def test_jsonl_invalid_line_reports_number():
    with pytest.raises(ParseError, match="line 2"):
        list(JsonlParser().parse('{"a": 1}\noops'))


def test_parse_error_is_value_error():
    assert issubclass(ParseError, ValueError)
