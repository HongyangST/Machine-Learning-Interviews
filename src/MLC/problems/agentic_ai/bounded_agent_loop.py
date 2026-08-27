"""A bounded tool-use loop with repeat-call detection.

Interview prompt: alternate between a decision policy and tool execution while
enforcing termination, step budget, and protection against repeated identical
calls. Keep orchestration control in code rather than only in the prompt.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class FinalAnswer:
    text: str


def run_agent_loop(
    decide: Callable[[list[Any]], ToolCall | FinalAnswer],
    tools: Mapping[str, Callable[..., Any]],
    *,
    max_steps: int = 8,
) -> tuple[str, list[Any]]:
    """Run until a final answer, rejecting unknown or repeated tool calls."""
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    trace: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for _ in range(max_steps):
        action = decide(trace.copy())
        trace.append(action)
        if isinstance(action, FinalAnswer):
            return action.text, trace
        signature = (action.name, repr(action.arguments))
        if signature in seen:
            raise RuntimeError("repeated identical tool call detected")
        seen.add(signature)
        if action.name not in tools:
            raise KeyError(f"unknown tool: {action.name}")
        result = tools[action.name](**dict(action.arguments))
        trace.append(result)
    raise RuntimeError("agent exceeded its step budget")

