"""TTS-Suite: engine containers the service starts on GPUs with room.

``/engines`` lists every engine with its status and ``vram_mib`` per GPU UUID. ``POST /release`` asks
the service to take engines down; it decides per engine and refuses one a reading holds (``in_use``).
There is no force.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from ..gpus import Gpu
from ..http_json import request_json
from ..model import Action, ConsumerError, Holder, Outcome

POLL_S = 2.0
HOLDING = ("running", "starting")


@dataclass
class TtsSuite:
    name: str
    url: str
    busy_wait_s: float = 0.0

    def holders(self, gpus: Sequence[Gpu]) -> list[Holder]:
        status, body = request_json("GET", f"{self.url}/engines")
        if status != 200 or not isinstance(body, list):
            raise ConsumerError(f"{self.name}: /engines answered HTTP {status}")
        holders = []
        for engine in body:
            # Engines on other hosts sit on other machines' cards.
            if engine.get("host") is not None or not engine.get("needs_gpu"):
                continue
            if engine.get("status") not in HOLDING:
                continue
            vram = engine.get("vram_mib")
            # None = the service could not measure where the engine sits: it may be on any card.
            cards = frozenset(gpu.uuid for gpu in gpus) if vram is None else frozenset(vram)
            holders.append(Holder(consumer=self.name, item=engine["engine"], gpu_uuids=cards, memory_mib=vram or {}))
        return holders

    def release(self, holders: Sequence[Holder], *, reason: str, deadline: float) -> list[Action]:
        started = time.monotonic()
        wait_until = min(started + self.busy_wait_s, deadline)
        pending = list(holders)
        actions: list[Action] = []
        while True:
            busy: list[Holder] = []
            for holder, (outcome, detail) in zip(pending, self._release(pending), strict=True):
                if outcome == "in_use":
                    busy.append(holder)
                elif outcome in ("stopped", "not_running"):
                    actions.append(Action(self.name, holder.item, "released", detail, self._waited(started)))
                else:
                    actions.append(Action(self.name, holder.item, "error", f"{outcome}: {detail}"))
            if not busy or time.monotonic() >= wait_until:
                break
            time.sleep(min(POLL_S, max(0.0, wait_until - time.monotonic())))
            pending = busy
        refused: Outcome = "timeout" if wait_until < started + self.busy_wait_s else "refused_busy"
        actions.extend(Action(self.name, h.item, refused, "a reading holds it", self._waited(started)) for h in busy)
        return actions

    def _release(self, holders: Sequence[Holder]) -> list[tuple[str, str]]:
        targets = [{"engine": holder.item, "host": None} for holder in holders]
        status, body = request_json("POST", f"{self.url}/release", {"targets": targets})
        if status != 200 or not isinstance(body, list):
            raise ConsumerError(f"{self.name}: /release answered HTTP {status}")
        by_engine = {result.get("engine"): (result.get("outcome", ""), result.get("detail", "")) for result in body}
        missing = [holder.item for holder in holders if holder.item not in by_engine]
        if missing:
            raise ConsumerError(f"{self.name}: /release did not answer for {missing}")
        return [by_engine[holder.item] for holder in holders]

    @staticmethod
    def _waited(started: float) -> float:
        return round(time.monotonic() - started, 1)
