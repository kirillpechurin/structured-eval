"""YamlParser — optional (`yaml` extra); skipped when PyYAML is absent."""

import sys

import pytest

from structured_eval.formats import YamlParser
from structured_eval.formats.base import ParseError

pytestmark = pytest.mark.unit


def test_parses_mapping() -> None:
    assert YamlParser().parse("a: 1\nb: two") == {"a": 1, "b": "two"}


def test_invalid_raises_parse_error() -> None:
    with pytest.raises(ParseError):
        YamlParser().parse("a: [1, 2\n  - broken")


def test_missing_extra_reports_the_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # PyYAML is imported at parse time, not at import time, so the core stays
    # installable without it — and asking for YAML anyway says what to install.
    monkeypatch.setitem(sys.modules, "yaml", None)

    with pytest.raises(ImportError, match="pip install pyyaml"):
        YamlParser().parse("a: 1")
