from typing import Any

from structured_eval.formats.base import ParseError


class YamlParser:
    """Parse a YAML string into a Python object.

    Uses yaml.safe_load — arbitrary Python object construction is disabled.
    Raises ParseError on malformed input. PyYAML is imported lazily so the
    core package stays importable without the ``yaml`` extra.
    """

    def parse(self, text: str) -> Any:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YAML parsing. Install it with: pip install pyyaml"
            ) from exc
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ParseError(f"Invalid YAML: {exc}") from exc
