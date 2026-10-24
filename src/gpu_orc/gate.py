"""One /free request at a time, first come first served, each waiting at most until its deadline."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class GateState:
    running: bool
    waiting: int
    running_since_s: float | None


class FifoGate:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._waiting: deque[object] = deque()
        self._running_since: float | None = None

    @contextmanager
    def enter(self, deadline: float) -> Iterator[bool]:
        """Yields True once it is this caller's turn, False when ``deadline`` (time.monotonic())
        passed while waiting."""
        ticket = object()
        with self._condition:
            self._waiting.append(ticket)
            while self._running_since is not None or self._waiting[0] is not ticket:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._waiting.remove(ticket)
                    self._condition.notify_all()
                    entered = False
                    break
                self._condition.wait(remaining)
            else:
                self._waiting.popleft()
                self._running_since = time.monotonic()
                entered = True
        try:
            yield entered
        finally:
            if entered:
                with self._condition:
                    self._running_since = None
                    self._condition.notify_all()

    def state(self) -> GateState:
        with self._condition:
            since = None if self._running_since is None else round(time.monotonic() - self._running_since, 1)
            return GateState(self._running_since is not None, len(self._waiting), since)
