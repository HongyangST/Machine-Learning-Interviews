"""Tests for bounded, permission-aware agentic coding answers."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from problems.agentic_ai.bounded_agent_loop import (  # noqa: E402
    FinalAnswer,
    ToolCall,
    run_agent_loop,
)
from problems.agentic_ai.planner_executor import execute_plan  # noqa: E402
from problems.agentic_ai.retry_with_idempotency import IdempotencyStore, retry  # noqa: E402
from problems.agentic_ai.sliding_window_memory import Message, select_memory  # noqa: E402
from problems.agentic_ai.tool_argument_validation import (  # noqa: E402
    validate_tool_arguments,
)
from problems.agentic_ai.tool_registry import ToolRegistry  # noqa: E402


class AgenticAIAnswersTest(unittest.TestCase):
    def test_tool_argument_validation(self):
        schema = {
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        }
        self.assertEqual(validate_tool_arguments({"city": "Paris"}, schema), {"city": "Paris"})
        with self.assertRaises(ValueError):
            validate_tool_arguments({}, schema)

    def test_tool_registry_permission_boundary(self):
        registry = ToolRegistry()
        registry.register("add", lambda left, right: left + right)
        registry.register("send", lambda message: message, side_effecting=True)
        self.assertEqual(registry.execute("add", {"left": 2, "right": 3}), 5)
        with self.assertRaises(PermissionError):
            registry.execute("send", {"message": "hello"})

    def test_planner_executor_stops_on_failure(self):
        def execute(step: str) -> str:
            if step == "fail":
                raise RuntimeError("nope")
            return step.upper()

        results = execute_plan(["first", "fail", "never"], execute)
        self.assertEqual([result.step for result in results], ["first", "fail"])
        self.assertFalse(results[-1].ok)

    def test_retry_and_idempotency(self):
        attempts = 0

        def flaky() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise TimeoutError("try again")
            return "ok"

        delays: list[float] = []
        self.assertEqual(retry(flaky, sleep=delays.append), "ok")
        self.assertEqual(delays, [0.1, 0.2])
        calls = 0

        def side_effect() -> int:
            nonlocal calls
            calls += 1
            return calls

        store = IdempotencyStore()
        self.assertEqual(store.run_once("request-1", side_effect), 1)
        self.assertEqual(store.run_once("request-1", side_effect), 1)
        self.assertEqual(calls, 1)

    def test_sliding_window_memory_pins_system_message(self):
        messages = [
            Message("system", "follow policy"),
            Message("user", "old request here"),
            Message("assistant", "recent answer"),
        ]
        selected = select_memory(messages, token_budget=6)
        self.assertEqual([message.role for message in selected], ["system", "assistant"])

    def test_bounded_agent_loop(self):
        def decide(trace: list[object]) -> ToolCall | FinalAnswer:
            if not trace:
                return ToolCall("double", (("value", 4),))
            return FinalAnswer(f"result={trace[-1]}")

        answer, trace = run_agent_loop(decide, {"double": lambda value: value * 2})
        self.assertEqual(answer, "result=8")
        self.assertEqual(len(trace), 3)


if __name__ == "__main__":
    unittest.main()

