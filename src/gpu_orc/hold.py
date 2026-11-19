"""The hold: somebody keeps this machine's cards for themselves (a game, other work) until they let go.

A hold has an owner and one of two states. REQUESTED: somebody asked for the cards (``POST /hold``); nothing
is touched yet. ACTIVE: the hold worker of the orchestrator cleared the cards once no work of others ran on them
any more, and ``GET /loading`` says so, so the callers that would put something new on the cards (a describer
of another machine, a TTS container started from elsewhere) leave them alone. GPU-Orc ends nothing itself:
what a caller loads without asking still loads. Only one hold exists at a time. The hold lives in a file, so
it outlasts a restart of GPU-Orc and of the machine; only ``DELETE /hold`` ends it.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

DEFAULT_HOLD_FILE = Path("~/.local/state/gpu-orc/hold.json")


class HoldFileError(ValueError):
    """The hold file exists but cannot be read as a hold."""


class HoldConflict(RuntimeError):
    """The hold belongs to somebody else (HTTP 409)."""

    def __init__(self, hold: Hold) -> None:
        super().__init__(f"the cards are held by {hold.owner!r} ({hold.state}, since {hold.since})")
        self.hold = hold


HoldState = Literal["requested", "active"]
REQUESTED: HoldState = "requested"
ACTIVE: HoldState = "active"
STATES: tuple[HoldState, ...] = (REQUESTED, ACTIVE)


@dataclass(frozen=True)
class Hold:
    owner: str
    reason: str
    since: str  # ISO 8601 with the offset of the machine
    state: HoldState
    free: dict[str, Any] | None = None  # the /free that cleared the cards, as it ended; None while requested

    def as_json(self) -> dict[str, Any]:
        return {
            "owner": self.owner, "reason": self.reason, "since": self.since, "state": self.state, "free": self.free,
        }


class HoldStore:
    """The hold of this machine, kept in ``path``. Raises HoldFileError when the file is unreadable."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._hold = self._read()

    def get(self) -> Hold | None:
        with self._lock:
            return self._hold

    def request(self, owner: str, reason: str) -> Hold:
        """A new hold of ``owner`` in the state REQUESTED, or its hold as it is when it has one already.
        Raises HoldConflict when somebody else holds the cards."""
        with self._lock:
            if self._hold is None:
                since = datetime.now().astimezone().isoformat(timespec="seconds")
                return self._write(Hold(owner, reason, since, REQUESTED))
            if self._hold.owner != owner:
                raise HoldConflict(self._hold)
            return self._hold

    def activate(self, expected: Hold, free: dict[str, Any]) -> Hold | None:
        """Turn the requested hold ``expected`` into an active one that keeps the result of clearing the cards,
        however it ended; None when ``expected`` is not the hold any more. The very object counts, not owner
        and reason: released and requested anew while the cards were cleared, it is another hold, and what
        was loaded in between is not in that result."""
        with self._lock:
            if self._hold is not expected or expected.state != REQUESTED:
                return None
            return self._write(replace(expected, state=ACTIVE, free=free))

    def release(self, owner: str, force: bool) -> Hold | None:
        """Let go of the hold (a requested one is withdrawn); returns the hold let go of, None when there
        was none. Raises HoldConflict when it belongs to somebody else and ``force`` is not set."""
        with self._lock:
            released = self._hold
            if released is not None and released.owner != owner and not force:
                raise HoldConflict(released)
            self._path.unlink(missing_ok=True)
            self._hold = None
            return released

    def _write(self, hold: Hold) -> Hold:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(".tmp")
        temporary.write_text(json.dumps(hold.as_json()) + "\n", encoding="utf-8")
        os.replace(temporary, self._path)
        self._hold = hold
        return hold

    def _read(self) -> Hold | None:
        try:
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as error:
            raise HoldFileError(f"cannot read the hold file {self._path}: {error}") from error
        try:
            data = json.loads(raw)
            state = data["state"]
            if state not in STATES:
                raise ValueError(f"unknown state {state!r}")
            free = data["free"]
            if free is not None and not isinstance(free, dict):
                raise ValueError("free is no object")
            return Hold(str(data["owner"]), str(data["reason"]), str(data["since"]), state, free)
        except (ValueError, KeyError, TypeError) as error:
            raise HoldFileError(f"the hold file {self._path} is malformed: {error!r}") from error
