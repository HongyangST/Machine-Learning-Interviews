"""Strict validation for a small subset of JSON tool schemas.

Interview prompt: reject malformed tool arguments before execution. A production
implementation should use a standards-compliant schema library and enforce the
same contract at both registration and invocation boundaries.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


TYPE_CHECKS: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def validate_tool_arguments(arguments: Mapping[str, Any], schema: Mapping[str, Any]) -> dict[str, Any]:
    """Validate required fields, primitive types, enums, and extra properties."""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    missing = required - arguments.keys()
    if missing:
        raise ValueError(f"missing required arguments: {sorted(missing)}")
    extras = arguments.keys() - properties.keys()
    if extras and schema.get("additionalProperties", True) is False:
        raise ValueError(f"unexpected arguments: {sorted(extras)}")
    result = dict(arguments)
    for name, value in result.items():
        specification = properties.get(name)
        if specification is None:
            continue
        expected_name = specification.get("type")
        expected = TYPE_CHECKS.get(expected_name)
        if expected is None:
            raise ValueError(f"unsupported schema type for {name}: {expected_name}")
        if expected_name in {"number", "integer"} and isinstance(value, bool):
            raise TypeError(f"{name} has the wrong type")
        if not isinstance(value, expected):
            raise TypeError(f"{name} has the wrong type")
        if "enum" in specification and value not in specification["enum"]:
            raise ValueError(f"{name} is not an allowed value")
    return result

