"""A permission-aware registry for local agent tools.

Interview prompt: register named callables, bind arguments safely, and prevent a
side-effecting tool from running unless the caller explicitly grants permission.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import inspect
from typing import Any


@dataclass(frozen=True)
class RegisteredTool:
    function: Callable[..., Any]
    side_effecting: bool = False


@dataclass
class ToolRegistry:
    _tools: dict[str, RegisteredTool] = field(default_factory=dict)

    def register(
        self, name: str, function: Callable[..., Any], *, side_effecting: bool = False
    ) -> None:
        if not name or name in self._tools:
            raise ValueError("tool name must be non-empty and unique")
        self._tools[name] = RegisteredTool(function, side_effecting)

    def execute(
        self, name: str, arguments: dict[str, Any], *, allow_side_effects: bool = False
    ) -> Any:
        """Validate the call signature and invoke one registered tool."""
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        tool = self._tools[name]
        if tool.side_effecting and not allow_side_effects:
            raise PermissionError(f"tool requires side-effect approval: {name}")
        bound = inspect.signature(tool.function).bind(**arguments)
        bound.apply_defaults()
        return tool.function(*bound.args, **bound.kwargs)

