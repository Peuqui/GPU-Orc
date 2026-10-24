import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import pytest
from fakes import GPUS

from gpu_orc.gpus import Gpu, GpuSpecError
from gpu_orc.model import Action, Consumer, ConsumerError, Holder, UnknownModel
from gpu_orc.orchestrator import FreeRequest, ModelServerUnreachable, NotFound, Orchestrator, RequestError
from gpu_orc.verify import VerifySettings

A, B, C = (gpu.uuid for gpu in GPUS)
FAST = VerifySettings(timeout_s=0.3, min_freed_mib=256, poll_s=0.05)
HOLDER_MIB = 1000


class Machine:
    """GPU memory that goes down when a fake service really lets go of what it held."""

    def __init__(self) -> None:
        self.used = {gpu.uuid: gpu.memory_used_mib + 5000 for gpu in GPUS}
        self.lock = threading.Lock()

    def query(self) -> list[Gpu]:
        with self.lock:
            return [replace(gpu, memory_used_mib=self.used[gpu.uuid]) for gpu in GPUS]

    def free(self, holder: Holder) -> None:
        with self.lock:
            for uuid in holder.gpu_uuids:
                self.used[uuid] -= holder.memory_mib.get(uuid, HOLDER_MIB)


@dataclass
class FakeService:
    name: str
    held: list[Holder] = field(default_factory=list)
    machine: Machine | None = None
    release_delay_s: float = 0.0
    released: list[str] = field(default_factory=list)
    fail: bool = False
    # Says "released" but keeps the holder (still reported) or keeps the memory (process stuck).
    keeps_holder: bool = False
    keeps_memory: bool = False

    def holders(self, gpus: Sequence[Gpu]) -> list[Holder]:
        if self.fail:
            raise ConsumerError(f"{self.name} is down")
        return list(self.held)

    def release(self, holders: Sequence[Holder], *, reason: str, deadline: float) -> list[Action]:
        time.sleep(self.release_delay_s)
        self.released.extend(h.item for h in holders)
        for holder in holders:
            if not self.keeps_holder:
                self.held.remove(holder)
            if not self.keeps_memory and self.machine is not None:
                self.machine.free(holder)
        return [Action(self.name, h.item, "released") for h in holders]


@dataclass
class FakeServer(FakeService):
    models: dict[str, frozenset[str]] = field(default_factory=dict)
    no_eviction: tuple[str, ...] = ()
    unreachable: bool = False

    def is_running(self, model: str) -> bool:
        if self.unreachable:
            raise ConsumerError("down")
        return any(h.item == model for h in self.held)

    def model_gpu_uuids(self, model: str, gpus: Sequence[Gpu]) -> frozenset[str]:
        if model == "broken":
            raise GpuSpecError("index without PCI_BUS_ID")
        if model not in self.models:
            raise UnknownModel(model)
        return self.models[model]

    def evicts_persistent(self, model: str) -> bool:
        return model not in self.no_eviction and not any(h.item == model and h.persistent for h in self.held)


MODELS = {"big": frozenset({A, B}), "small-vlm": frozenset({B}), "elsewhere": frozenset({C})}


def model(item: str, *cards: str, persistent: bool = False) -> Holder:
    return Holder("ls", item, frozenset(cards), persistent=persistent)


def service(name: str, *held: Holder, machine: Machine | None = None, **kwargs: Any) -> FakeService:
    return FakeService(name, list(held), machine, **kwargs)


def server(*held: Holder, machine: Machine | None = None, **kwargs: Any) -> FakeServer:
    return FakeServer("ls", list(held), machine, models=MODELS, **kwargs)


def make(*consumers: Consumer, machine: Machine | None = None, timeout: float = 30.0) -> Orchestrator:
    query = machine.query if machine is not None else lambda: list(GPUS)
    return Orchestrator(consumers, timeout, query=query, verify_settings=FAST)


def outcomes(result: dict[str, Any]) -> list[tuple[str, str, str]]:
    return [(a["consumer"], a["item"], a["outcome"]) for a in result["actions"]]


def in_background(orchestrator: Orchestrator, request: FreeRequest) -> threading.Thread:
    thread = threading.Thread(target=orchestrator.free, args=(request,))
    thread.start()
    return thread


def test_model_form_frees_services_and_persistent_models_on_its_cards() -> None:
    m = Machine()
    stt = service("stt", Holder("stt", "gpu-worker", frozenset({B}), memory_mib={B: 3000}), machine=m)
    tts = service("tts", Holder("tts", "qwen", frozenset({C})), machine=m)
    ls = server(model("main-old", A, B), model("describer", B, persistent=True), model("embed", C, persistent=True),
                machine=m)
    result = make(stt, tts, ls, machine=m).free(FreeRequest(reason="t", model="big"))
    assert outcomes(result) == [("stt", "gpu-worker", "released"), ("ls", "describer", "released")]
    # The server swaps its other models itself; the card the model does not use stays as it was.
    assert tts.released == [] and "main-old" not in ls.released and "embed" not in ls.released
    assert result["free"] is True
    memory = {entry["index"]: (entry["before_mib"], entry["after_mib"]) for entry in result["memory"]}
    assert memory == {0: (6000, 6000), 1: (5500, 1500)}  # card 1: −3000 worker, −1000 describer


def test_ok_but_still_reported_is_not_freed() -> None:
    m = Machine()
    stt = service("stt", Holder("stt", "gpu-worker", frozenset({A})), machine=m, keeps_holder=True)
    result = make(stt, server(), machine=m).free(FreeRequest(reason="t", gpus=(0,)))
    action = result["actions"][0]
    assert action["outcome"] == "not_freed" and "still reported" in action["detail"]
    assert result["free"] is False


def test_ok_but_memory_kept_is_not_freed() -> None:
    m = Machine()
    stt = service("stt", Holder("stt", "gpu-worker", frozenset({A}), memory_mib={A: 3000}), machine=m,
                  keeps_memory=True)
    result = make(stt, server(), machine=m).free(FreeRequest(reason="t", gpus=(0,)))
    action = result["actions"][0]
    assert action["outcome"] == "not_freed"
    assert "no longer reported" in action["detail"] and "6000 → 6000 MiB, expected −1500" in action["detail"]
    assert result["free"] is False


def test_memory_returned_late_by_the_driver_still_passes() -> None:
    m = Machine()

    class Lagging(FakeService):
        def release(self, holders: Sequence[Holder], *, reason: str, deadline: float) -> list[Action]:
            for holder in holders:
                self.held.remove(holder)
                threading.Timer(0.15, m.free, args=(holder,)).start()
            return [Action(self.name, h.item, "released") for h in holders]

    lagging = Lagging("stt", [Holder("stt", "gpu-worker", frozenset({A}))])
    result = make(lagging, server(), machine=m).free(FreeRequest(reason="t", gpus=(0,)))
    assert outcomes(result) == [("stt", "gpu-worker", "released")] and result["free"] is True


def test_model_that_evicts_nothing_leaves_persistent_models() -> None:
    m = Machine()
    ls = server(model("describer", B, persistent=True), machine=m, no_eviction=("small-vlm",))
    stt = service("stt", Holder("stt", "gpu-worker", frozenset({B})), machine=m)
    result = make(stt, ls, machine=m).free(FreeRequest(reason="t", model="small-vlm"))
    assert outcomes(result) == [("stt", "gpu-worker", "released")]
    assert ls.released == []


def test_already_running_skips_the_queue() -> None:
    m = Machine()
    slow = service("stt", Holder("stt", "gpu-worker", frozenset({A})), machine=m, release_delay_s=1.0)
    orchestrator = make(slow, server(model("big", A, B), machine=m), machine=m)
    blocker = in_background(orchestrator, FreeRequest(reason="block", gpus=(0,), keep=("ls",)))
    time.sleep(0.2)
    started = time.monotonic()
    result = orchestrator.free(FreeRequest(reason="hook", model="big"))
    assert result["already_running"] is True and result["actions"] == []
    assert time.monotonic() - started < 0.5
    blocker.join()


def test_gpus_form_frees_main_models_too_unless_kept() -> None:
    m = Machine()
    ls = server(model("main", A), model("other", B), machine=m)
    result = make(ls, machine=m).free(FreeRequest(reason="t", gpus=(0, "GPU-bbbb"), keep=("ls:other",)))
    assert outcomes(result) == [("ls", "main", "released"), ("ls", "other", "kept")]
    assert result["free"] is True


def test_all_gpus() -> None:
    m = Machine()
    stt = service("stt", Holder("stt", "gpu-worker", frozenset({C})), machine=m)
    result = make(stt, server(), machine=m).free(FreeRequest(reason="t", gpus="all"))
    assert outcomes(result) == [("stt", "gpu-worker", "released")]
    assert len(result["status"]["gpus"]) == 3 and len(result["memory"]) == 3


def test_dry_run_touches_nothing() -> None:
    stt = service("stt", Holder("stt", "gpu-worker", frozenset({A})))
    result = make(stt, server()).free(FreeRequest(reason="t", gpus=(0,), dry_run=True))
    assert outcomes(result) == [("stt", "gpu-worker", "would_release")]
    assert stt.released == [] and result["free"] is False
    assert result["memory"] == [{"index": 0, "uuid": A, "before_mib": 1000, "after_mib": None}]


def test_consumers_are_asked_in_parallel() -> None:
    m = Machine()
    slow_a = service("a", Holder("a", "x", frozenset({A})), machine=m, release_delay_s=0.5)
    slow_b = service("b", Holder("b", "y", frozenset({A})), machine=m, release_delay_s=0.5)
    started = time.monotonic()
    make(slow_a, slow_b, server(), machine=m).free(FreeRequest(reason="t", gpus=(0,)))
    assert time.monotonic() - started < 0.9


def test_queue_wait_counts_against_the_overall_limit() -> None:
    m = Machine()
    slow = service("stt", Holder("stt", "gpu-worker", frozenset({A})), machine=m, release_delay_s=1.0)
    other = service("tts", Holder("tts", "qwen", frozenset({B})), machine=m)
    orchestrator = make(slow, other, server(), machine=m, timeout=0.5)
    blocker = in_background(orchestrator, FreeRequest(reason="block", gpus=(0,)))
    time.sleep(0.1)
    result = orchestrator.free(FreeRequest(reason="late", gpus=(1,)))
    blocker.join()
    assert outcomes(result) == [("tts", "qwen", "timeout")]
    assert other.released == []


def test_consumer_that_cannot_report_is_an_error_and_not_free() -> None:
    result = make(service("stt", fail=True), server()).free(FreeRequest(reason="t", gpus=(0,)))
    assert outcomes(result) == [("stt", "", "error")]
    assert result["free"] is False


def test_a_kept_consumer_that_cannot_report_does_not_matter() -> None:
    # The calibration stops the model server, then has everything else cleared and the server kept.
    m = Machine()
    stt = service("stt", Holder("stt", "gpu-worker", frozenset({A})), machine=m)
    result = make(stt, server(fail=True), machine=m).free(FreeRequest(reason="t", gpus="all", keep=("ls",)))
    assert outcomes(result) == [("stt", "gpu-worker", "released")]
    assert result["free"] is True
    assert result["status"]["errors"] == [{"consumer": "ls", "detail": "ls is down"}]


def test_keeping_one_item_does_not_hide_its_consumers_error() -> None:
    result = make(service("stt", fail=True), server()).free(FreeRequest(reason="t", gpus=(0,), keep=("stt:gpu-worker",)))
    assert outcomes(result) == [("stt", "", "error")]
    assert result["free"] is False


@pytest.mark.parametrize("request_, error", [
    (FreeRequest(reason="t", model="ghost"), RequestError),
    (FreeRequest(reason="t", model="broken"), RequestError),
    (FreeRequest(reason="t", gpus=(9,)), RequestError),
    (FreeRequest(reason="t", gpus=(0,), keep=("nope",)), RequestError),
    (FreeRequest(reason="t", model="big", model_server="vllm"), NotFound),
])
def test_bad_requests(request_: FreeRequest, error: type[Exception]) -> None:
    with pytest.raises(error):
        make(service("stt"), server()).free(request_)


def test_model_server_required_when_several() -> None:
    with pytest.raises(RequestError, match="model_server"):
        make(server(), FakeServer("other")).free(FreeRequest(reason="t", model="big"))


def test_unreachable_model_server() -> None:
    with pytest.raises(ModelServerUnreachable):
        make(server(unreachable=True)).free(FreeRequest(reason="t", model="big"))


def test_status_attributes_memory_only_where_reported() -> None:
    tts = service("tts", Holder("tts", "qwen", frozenset({A}), memory_mib={A: 600}))
    status = make(tts, server(model("main", B))).status()
    by_index = {g["index"]: g for g in status["gpus"]}
    assert by_index[0]["unattributed_mib"] == 400      # 1000 used, 600 reported
    assert by_index[1]["unattributed_mib"] is None     # llama-swap reports no memory
    assert by_index[2]["unattributed_mib"] == 0 and by_index[2]["holders"] == []
    assert status["queue"] == {"running": False, "waiting": 0, "running_since_s": None}
