"""STT-Suite: a GPU worker (and a diarization worker) the service starts on demand.

``/status`` names the card of each worker by UUID. ``POST /unload?device=cuda`` ends both workers;
while a transcription runs the service answers 409, ``force=1`` ends it regardless.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from ..gpus import Gpu
from ..http_json import request_json
from ..model import Action, ConsumerError, Holder, Outcome

HTTP_CONFLICT = 409


@dataclass(frozen=True)
class _WorkerFields:
    loaded: str
    uuid: str
    footprint: str
    busy: str


WORKERS = (
    ("gpu-worker", _WorkerFields("gpu_loaded", "gpu_uuid", "gpu_footprint_mib", "gpu_busy")),
    ("diarize-worker", _WorkerFields("diarize_loaded", "diarize_gpu_uuid", "diarize_footprint_mib", "diarize_busy")),
)
POLL_S = 2.0
# The forcing request itself must still fit before the overall deadline.
FORCE_MARGIN_S = 5.0


@dataclass
class SttSuite:
    name: str
    url: str
    busy_wait_s: float = 0.0
    force_after_wait: bool = False

    def holders(self, gpus: Sequence[Gpu]) -> list[Holder]:
        status, body = request_json("GET", f"{self.url}/status")
        if status != 200 or not isinstance(body, dict):
            raise ConsumerError(f"{self.name}: /status answered HTTP {status}")
        holders = []
        for item, fields in WORKERS:
            if not body.get(fields.loaded):
                continue
            uuid = body.get(fields.uuid)
            if not uuid:
                raise ConsumerError(
                    f"{self.name}: /status reports a running {item} without {fields.uuid} "
                    "(STT-Suite too old for GPU-Orc)"
                )
            footprint = body.get(fields.footprint) or 0
            holders.append(Holder(
                consumer=self.name,
                item=item,
                gpu_uuids=frozenset({uuid}),
                memory_mib={uuid: footprint} if footprint else {},
                busy=bool(body.get(fields.busy)),
            ))
        return holders

    def release(self, holders: Sequence[Holder], *, reason: str, deadline: float) -> list[Action]:
        started = time.monotonic()
        wait_until = min(started + self.busy_wait_s, deadline - FORCE_MARGIN_S)
        while True:
            status, _ = request_json("POST", f"{self.url}/unload?device=cuda")
            if status == 200:
                return self._actions(holders, "released", started)
            if status != HTTP_CONFLICT:
                raise ConsumerError(f"{self.name}: /unload answered HTTP {status}")
            if time.monotonic() >= wait_until:
                break
            time.sleep(min(POLL_S, max(0.0, wait_until - time.monotonic())))
        if self.force_after_wait:
            status, _ = request_json("POST", f"{self.url}/unload?device=cuda&force=1")
            if status != 200:
                raise ConsumerError(f"{self.name}: /unload with force answered HTTP {status}")
            return self._actions(holders, "released_forced", started)
        # Busy past its own patience = the service may not be forced; past the overall limit = timeout.
        outcome: Outcome = "timeout" if wait_until < started + self.busy_wait_s else "refused_busy"
        return self._actions(holders, outcome, started, "transcription still running")

    def _actions(
        self, holders: Sequence[Holder], outcome: Outcome, started: float, detail: str = ""
    ) -> list[Action]:
        waited = round(time.monotonic() - started, 1)
        return [Action(self.name, holder.item, outcome, detail, waited) for holder in holders]
