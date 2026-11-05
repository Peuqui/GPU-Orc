"""The TTS containers that the TTS-Suite of another machine runs on this one (its hosts, started over SSH).

That suite speaks for them: ``GET /engines`` names its catalog rows with their host and status, and
``POST /release`` with targets ``{engine, host}`` stops a container on a host (it never stops one a reading
holds: ``in_use``). The suite listens beyond its own loopback here, so every request carries its token
(``X-TTS-Suite-Token``). It does not report which card of this machine a container uses or how much it
holds there, so a running container counts as holding every card, with its memory unknown.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from ..gpus import Gpu
from ..http_json import request_json
from ..model import Action, ConsumerError, Holder

TOKEN_HEADER = "X-TTS-Suite-Token"
POLL_S = 2.0
HOLDING = ("running", "starting")


@dataclass
class TtsSuiteRemote:
    name: str
    url: str  # the TTS-Suite on the machine that controls the containers
    host: str  # this machine's name in that suite's hosts
    token: str
    busy_wait_s: float = 0.0  # how long to wait for a reading that holds a container

    def holders(self, gpus: Sequence[Gpu]) -> list[Holder]:
        status, body = request_json("GET", f"{self.url}/engines", headers={TOKEN_HEADER: self.token})
        if status != 200 or not isinstance(body, list):
            raise ConsumerError(f"{self.name}: /engines answered HTTP {status}")
        cards = frozenset(gpu.uuid for gpu in gpus)
        return [
            Holder(consumer=self.name, item=str(row["engine"]), gpu_uuids=cards)
            for row in body
            if isinstance(row, dict) and row.get("host") == self.host and row.get("status") in HOLDING
        ]

    def release(self, holders: Sequence[Holder], *, reason: str, deadline: float) -> list[Action]:
        started = time.monotonic()
        wait_until = min(started + self.busy_wait_s, deadline)
        pending = [holder.item for holder in holders]
        actions: list[Action] = []
        while True:
            busy = []
            for engine, (outcome, detail) in self._ask(pending, deadline).items():
                if outcome in ("stopped", "not_running"):
                    actions.append(Action(self.name, engine, "released", detail, _waited(started)))
                elif outcome == "in_use":
                    busy.append(engine)
                else:  # unreachable, failed
                    actions.append(Action(self.name, engine, "not_freed", f"service: {outcome}: {detail}"))
            if not busy or time.monotonic() >= wait_until:
                break
            time.sleep(min(POLL_S, max(0.0, wait_until - time.monotonic())))
            pending = busy
        actions.extend(Action(self.name, engine, "refused_busy", "a reading holds it", _waited(started))
                       for engine in busy)
        return actions

    def _ask(self, engines: Sequence[str], deadline: float) -> dict[str, tuple[str, str]]:
        body = {"targets": [{"engine": engine, "host": self.host} for engine in engines]}
        timeout = max(1.0, deadline - time.monotonic())
        status, answer = request_json("POST", f"{self.url}/release", body, timeout=timeout,
                                      headers={TOKEN_HEADER: self.token})
        if status != 200 or not isinstance(answer, list):
            raise ConsumerError(f"{self.name}: /release answered HTTP {status}")
        by_engine = {
            str(r.get("engine")): (str(r.get("outcome", "")), str(r.get("detail", "")))
            for r in answer if isinstance(r, dict) and r.get("host") == self.host
        }
        missing = [engine for engine in engines if engine not in by_engine]
        if missing:
            raise ConsumerError(f"{self.name}: /release did not answer for {missing}")
        return {engine: by_engine[engine] for engine in engines}


def _waited(started: float) -> float:
    return round(time.monotonic() - started, 1)
