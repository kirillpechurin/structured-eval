from typing import Any

from structured_eval.parsers.base import ParseError

try:
    import yaml
    from yaml import YAMLError
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyYAML is required for YAML parsing. Install it with: pip install pyyaml"
    ) from exc


class YamlParser:
    """Parse a YAML string into a Python object.

    Uses yaml.safe_load — arbitrary Python object construction is disabled.
    Raises ParseError on malformed input.
    """

    def parse(self, text: str) -> Any:
        try:
            return yaml.safe_load(text)
        except YAMLError as exc:
            raise ParseError(f"Invalid YAML: {exc}") from exc
