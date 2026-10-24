"""A service with the suites' common GPU release API (STT-Suite, TTS-Suite).

``GET /gpu/holders`` names what the service holds on this machine's GPUs, the cards taken from the
process's configuration. ``POST /gpu/release`` takes ``items``, ``force`` and ``reason`` and answers one
result per item: ``released`` (the process with the CUDA context has ended, proven by the service),
``not_running``, ``busy`` (left alone, only without force) or ``failed`` (the end could not be proven).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from ..gpus import Gpu
from ..http_json import request_json
from ..model import Action, ConsumerError, Holder, Outcome

POLL_S = 2.0
# The forcing request itself must still fit before the overall deadline.
FORCE_MARGIN_S = 5.0


@dataclass
class GpuRelease:
    name: str
    url: str
    busy_wait_s: float = 0.0
    force_after_wait: bool = False

    def holders(self, gpus: Sequence[Gpu]) -> list[Holder]:
        status, body = request_json("GET", f"{self.url}/gpu/holders")
        if status != 200 or not isinstance(body, dict) or not isinstance(body.get("holders"), list):
            raise ConsumerError(f"{self.name}: /gpu/holders answered HTTP {status}")
        known = {gpu.uuid for gpu in gpus}
        holders = []
        for entry in body["holders"]:
            cards = frozenset(entry.get("gpu_uuids") or [])
            unknown = cards - known
            if unknown:
                raise ConsumerError(f"{self.name}: {entry.get('item')} names GPUs not on this machine: {sorted(unknown)}")
            holders.append(Holder(
                consumer=self.name,
                item=str(entry["item"]),
                gpu_uuids=cards,
                memory_mib=entry.get("memory_mib") or {},
                busy=entry.get("busy") if isinstance(entry.get("busy"), bool) else None,
            ))
        return holders

    def release(self, holders: Sequence[Holder], *, reason: str, deadline: float) -> list[Action]:
        started = time.monotonic()
        wait_until = min(started + self.busy_wait_s, deadline - FORCE_MARGIN_S)
        pending = [holder.item for holder in holders]
        actions: list[Action] = []
        while True:
            busy = self._settle(self._ask(pending, False, reason), "released", started, actions)
            if not busy or time.monotonic() >= wait_until:
                break
            time.sleep(min(POLL_S, max(0.0, wait_until - time.monotonic())))
            pending = busy
        if busy and self.force_after_wait:
            busy = self._settle(self._ask(busy, True, reason), "released_forced", started, actions)
        # Busy past its own patience = it may not be forced; cut short by the overall limit = timeout.
        left: Outcome = "timeout" if wait_until < started + self.busy_wait_s else "refused_busy"
        actions.extend(Action(self.name, item, left, "still busy", _waited(started)) for item in busy)
        return actions

    def _ask(self, items: Sequence[str], force: bool, reason: str) -> dict[str, tuple[str, str]]:
        body = {"items": list(items), "force": force, "reason": reason}
        status, answer = request_json("POST", f"{self.url}/gpu/release", body)
        results = answer.get("results") if isinstance(answer, dict) else None
        if status not in (200, 409, 500) or not isinstance(results, list):
            raise ConsumerError(f"{self.name}: /gpu/release answered HTTP {status}")
        by_item = {str(r.get("item")): (str(r.get("outcome", "")), str(r.get("detail", ""))) for r in results}
        missing = [item for item in items if item not in by_item]
        if missing:
            raise ConsumerError(f"{self.name}: /gpu/release did not answer for {missing}")
        return {item: by_item[item] for item in items}

    def _settle(
        self, answers: dict[str, tuple[str, str]], done: Outcome, started: float, actions: list[Action]
    ) -> list[str]:
        """Turn the answers into actions; return the items still busy."""
        busy = []
        for item, (outcome, detail) in answers.items():
            if outcome in ("released", "not_running"):
                actions.append(Action(self.name, item, done, detail, _waited(started)))
            elif outcome == "busy":
                busy.append(item)
            elif outcome == "failed":
                actions.append(Action(self.name, item, "not_freed", f"service: {detail}", _waited(started)))
            else:
                actions.append(Action(self.name, item, "error", f"unknown outcome {outcome!r}: {detail}"))
        return busy


def _waited(started: float) -> float:
    return round(time.monotonic() - started, 1)
