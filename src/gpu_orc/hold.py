"""The hold: somebody keeps this machine's cards for themselves (a game, other work) until they let go.

While held, ``GET /loading`` and ``GET /status`` say so, and the callers that would put something new on the
cards (a describer of another machine, a TTS container started from elsewhere) ask there first and leave
them alone. Setting the hold also clears every card at once (``POST /hold``). GPU-Orc ends nothing itself:
what a caller loads without asking still loads. The hold lives in a file, so it outlasts a restart of
GPU-Orc and of the machine; only ``DELETE /hold`` ends it.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DEFAULT_HOLD_FILE = Path("~/.local/state/gpu-orc/hold.json")


class HoldFileError(ValueError):
    """The hold file exists but cannot be read as a hold."""


@dataclass(frozen=True)
class Hold:
    since: str  # ISO 8601 with the offset of the machine
    reason: str

    def as_json(self) -> dict[str, str]:
        return {"since": self.since, "reason": self.reason}


class HoldStore:
    """The hold of this machine, kept in ``path``. Raises HoldFileError when the file is unreadable."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._hold = self._read()

    def get(self) -> Hold | None:
        with self._lock:
            return self._hold

    def set(self, reason: str) -> Hold:
        """Hold the cards (a hold already set keeps its time and gets the new reason)."""
        with self._lock:
            since = self._hold.since if self._hold else datetime.now().astimezone().isoformat(timespec="seconds")
            hold = Hold(since, reason)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._path.with_suffix(".tmp")
            temporary.write_text(json.dumps(hold.as_json()) + "\n", encoding="utf-8")
            os.replace(temporary, self._path)
            self._hold = hold
            return hold

    def clear(self) -> None:
        with self._lock:
            self._path.unlink(missing_ok=True)
            self._hold = None

    def _read(self) -> Hold | None:
        try:
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as error:
            raise HoldFileError(f"cannot read the hold file {self._path}: {error}") from error
        try:
            data = json.loads(raw)
            return Hold(str(data["since"]), str(data["reason"]))
        except (ValueError, KeyError, TypeError) as error:
            raise HoldFileError(f"the hold file {self._path} is malformed: {error!r}") from error
