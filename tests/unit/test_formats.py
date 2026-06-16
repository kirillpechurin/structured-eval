"""Unit tests for the format parsers (JSON / JSONL / YAML) and ParseError."""

from __future__ import annotations

import pytest

from structured_eval.formats.base import ParseError
from structured_eval.formats.json_parser import JsonlParser, JsonParser

pytestmark = pytest.mark.unit


class TestJson:
    def test_object(self):
        assert JsonParser().parse('{"a": 1}') == {"a": 1}

    def test_array(self):
        assert JsonParser().parse("[1, 2]") == [1, 2]

    def test_scalar(self):
        assert JsonParser().parse("42") == 42

    def test_invalid_raises_parse_error(self):
        with pytest.raises(ParseError):
            JsonParser().parse("{not json}")


class TestJsonl:
    def test_lines(self):
        out = list(JsonlParser().parse('{"a": 1}\n{"a": 2}'))
        assert out == [{"a": 1}, {"a": 2}]

    def test_blank_lines_skipped(self):
        out = list(JsonlParser().parse('{"a": 1}\n\n{"a": 2}\n'))
        assert out == [{"a": 1}, {"a": 2}]

    def test_invalid_line_reports_number(self):
        with pytest.raises(ParseError, match="line 2"):
            list(JsonlParser().parse('{"a": 1}\noops'))


class TestYaml:
    def test_parses_mapping(self):
        yaml_parser = pytest.importorskip("structured_eval.formats.yaml_parser")
        parser = yaml_parser.YamlParser()
        assert parser.parse("a: 1\nb: two") == {"a": 1, "b": "two"}

    def test_invalid_raises(self):
        yaml_parser = pytest.importorskip("structured_eval.formats.yaml_parser")
        with pytest.raises(ParseError):
            yaml_parser.YamlParser().parse("a: [1, 2\n  - broken")


def test_parse_error_is_value_error():
    assert issubclass(ParseError, ValueError)
