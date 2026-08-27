"""Conversation memory with a simple token budget and pinned system message.

Interview prompt: keep recent messages under a context budget without dropping
the system instruction. Production systems replace whitespace counts with the
model tokenizer and may summarize evicted history.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    role: str
    content: str


def select_memory(messages: Sequence[Message], token_budget: int) -> list[Message]:
    """Keep the first system message and as much recent history as fits."""
    if token_budget < 0:
        raise ValueError("token_budget must be non-negative")
    system = next((message for message in messages if message.role == "system"), None)
    selected: list[Message] = []
    remaining = token_budget
    if system is not None:
        cost = _token_count(system)
        if cost > remaining:
            raise ValueError("system message alone exceeds the token budget")
        selected.append(system)
        remaining -= cost
    recent: list[Message] = []
    for message in reversed(messages):
        if message is system:
            continue
        cost = _token_count(message)
        if cost > remaining:
            break
        recent.append(message)
        remaining -= cost
    return selected + list(reversed(recent))


def _token_count(message: Message) -> int:
    return len(message.content.split()) + 1

