"""A deterministic planner/executor boundary with structured results.

Interview prompt: separate planning from execution, record every step, and stop
on failure without hiding partial progress. Real agents also persist traces and
re-plan only under an explicit policy.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StepResult:
    step: str
    ok: bool
    output: Any = None
    error: str | None = None


def execute_plan(
    steps: Sequence[str],
    executor: Callable[[str], Any],
    *,
    max_steps: int = 10,
) -> list[StepResult]:
    """Execute a bounded plan in order and stop at the first failed step."""
    if max_steps < 1 or len(steps) > max_steps:
        raise ValueError("plan exceeds the allowed step budget")
    results: list[StepResult] = []
    for step in steps:
        try:
            results.append(StepResult(step=step, ok=True, output=executor(step)))
        except Exception as error:  # The boundary turns tool errors into trace data.
            results.append(StepResult(step=step, ok=False, error=str(error)))
            break
    return results

