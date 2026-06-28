"""YamlParser — optional (``yaml`` extra); skipped when PyYAML is absent."""

import pytest

from structured_eval.formats.base import ParseError

pytestmark = pytest.mark.unit

yaml_parser = pytest.importorskip("structured_eval.formats.yaml_parser")


def test_parses_mapping():
    assert yaml_parser.YamlParser().parse("a: 1\nb: two") == {"a": 1, "b": "two"}


def test_invalid_raises_parse_error():
    with pytest.raises(ParseError):
        yaml_parser.YamlParser().parse("a: [1, 2\n  - broken")
