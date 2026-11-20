"""Work in progress: somebody tells GPU-Orc that a job runs on this machine, so that whoever would restart or
switch the machine off (the weekly reboot) can ask first and leave it alone.

A report is a lease: it ends with ``end``, or by itself when ``ttl_s`` passes without a renewal (a caller that
crashed ends nothing and must not block the machine for ever). The same owner reporting again renews its
lease. The reports live in memory only: a restart of the machine ends the jobs as well, and after a restart of
GPU-Orc the next renewal brings every running job back.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Work:
    owner: str
    reason: str
    started_s: float  # time.monotonic() of the first report
    expires_s: float  # time.monotonic() after which the lease is over

    def as_json(self, now_s: float) -> dict[str, str | float]:
        return {
            "owner": self.owner,
            "reason": self.reason,
            "running_for_s": round(now_s - self.started_s, 1),
            "expires_in_s": round(self.expires_s - now_s, 1),
        }


class WorkStore:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._work: dict[str, Work] = {}

    def begin(self, owner: str, reason: str, ttl_s: float) -> None:
        """Report work, or renew the lease of an owner that reports already (it keeps its start, takes the new reason)."""
        with self._lock:
            now = self._clock()
            self._drop_expired(now)
            earlier = self._work.get(owner)
            self._work[owner] = Work(owner, reason, earlier.started_s if earlier else now, now + ttl_s)

    def end(self, owner: str) -> bool:
        """End the work of ``owner``; False when it had none (it expired, or GPU-Orc restarted in between)."""
        with self._lock:
            self._drop_expired(self._clock())
            return self._work.pop(owner, None) is not None

    def active(self) -> list[dict[str, str | float]]:
        with self._lock:
            now = self._clock()
            self._drop_expired(now)
            return [work.as_json(now) for work in self._work.values()]

    def _drop_expired(self, now: float) -> None:
        self._work = {owner: work for owner, work in self._work.items() if work.expires_s > now}

    def others_running(self, owner: str) -> bool:
        """Whether work of anybody but ``owner`` is reported (the work of ``owner`` runs on the cards it holds)."""
        with self._lock:
            self._drop_expired(self._clock())
            return any(work_owner != owner for work_owner in self._work)
