"""Bounded exponential retry plus an in-memory idempotency guard.

Interview prompt: retry transient failures without duplicating a side effect.
Production implementations persist idempotency results in durable shared storage.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar


T = TypeVar("T")


def retry(
    operation: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay: float = 0.1,
    sleep: Callable[[float], None] = lambda _delay: None,
    retry_on: tuple[type[Exception], ...] = (TimeoutError, ConnectionError),
) -> T:
    """Retry only declared transient exceptions with exponential backoff."""
    if max_attempts < 1 or base_delay < 0:
        raise ValueError("invalid retry policy")
    for attempt in range(max_attempts):
        try:
            return operation()
        except retry_on:
            if attempt + 1 == max_attempts:
                raise
            sleep(base_delay * (2**attempt))
    raise AssertionError("unreachable")


@dataclass
class IdempotencyStore:
    _results: dict[str, Any] = field(default_factory=dict)

    def run_once(self, key: str, operation: Callable[[], T]) -> T:
        """Return the recorded result when the same idempotency key repeats."""
        if not key:
            raise ValueError("idempotency key must be non-empty")
        if key not in self._results:
            self._results[key] = operation()
        return self._results[key]

