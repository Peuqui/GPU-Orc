"""The machine the client tests run against: the real GpuOrcServer with fake services and fake memory."""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from fakes import GPUS
from test_orchestrator import FAST, MODELS, FakeServer, FakeService, Machine, model, server, service

from gpu_orc.hold import HoldStore
from gpu_orc.model import Action, Holder
from gpu_orc.orchestrator import Orchestrator
from gpu_orc.work import WorkStore

A, B, C = (gpu.uuid for gpu in GPUS)


@dataclass
class Stubborn(FakeService):
    """A service whose item stays busy: it refuses to let go."""

    def release(self, holders: Sequence[Holder], *, reason: str, deadline: float) -> list[Action]:
        return [Action(self.name, h.item, "refused_busy", "still busy") for h in holders]


def machine(*, stubborn: bool = False, two_servers: bool = False) -> Orchestrator:
    """stt holds a worker on card B, "ls" runs the main model "big" on card A. With ``stubborn``, tts holds an
    engine on card C that refuses to let go; the model "elsewhere" lives there."""
    mem = Machine()
    consumers: list[FakeService] = [
        service("stt", Holder("stt", "gpu-worker", frozenset({B}), memory_mib={B: 3000}), machine=mem),
        server(model("big", A), machine=mem),
    ]
    if stubborn:
        consumers.append(Stubborn("tts", [Holder("tts", "qwen", frozenset({C}))], mem))
    if two_servers:
        consumers.append(FakeServer("vllm", [], mem, models=MODELS))
    return Orchestrator(consumers, 30, query=mem.query, verify_settings=FAST, hold=HoldStore(Path(tempfile.mkdtemp()) / "hold.json"),
                        work=WorkStore())
