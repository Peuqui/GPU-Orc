import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from fakes import GPUS

from gpu_orc.gpus import Gpu, GpuSpecError
from gpu_orc.model import Action, Consumer, ConsumerError, Holder, UnknownModel
from gpu_orc.orchestrator import FreeRequest, ModelServerUnreachable, NotFound, Orchestrator, RequestError

A, B, C = (gpu.uuid for gpu in GPUS)


@dataclass
class FakeService:
    name: str
    held: list[Holder] = field(default_factory=list)
    release_delay_s: float = 0.0
    released: list[str] = field(default_factory=list)
    fail: bool = False

    def holders(self, gpus: Sequence[Gpu]) -> list[Holder]:
        if self.fail:
            raise ConsumerError(f"{self.name} is down")
        return list(self.held)

    def release(self, holders: Sequence[Holder], *, reason: str, deadline: float) -> list[Action]:
        time.sleep(self.release_delay_s)
        self.released.extend(h.item for h in holders)
        self.held = [h for h in self.held if h not in holders]
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


def server(*held: Holder, no_eviction: tuple[str, ...] = (), unreachable: bool = False) -> FakeServer:
    models = {"big": frozenset({A, B}), "small-vlm": frozenset({B}), "elsewhere": frozenset({C})}
    return FakeServer("ls", list(held), models=models, no_eviction=no_eviction, unreachable=unreachable)


def model(item: str, *cards: str, persistent: bool = False) -> Holder:
    return Holder("ls", item, frozenset(cards), persistent=persistent)


def make(*consumers: Consumer, timeout: float = 30.0) -> Orchestrator:
    return Orchestrator(consumers, timeout, query=lambda: list(GPUS))


def outcomes(result: dict[str, Any]) -> list[tuple[str, str, str]]:
    return [(a["consumer"], a["item"], a["outcome"]) for a in result["actions"]]


def in_background(orchestrator: Orchestrator, request: FreeRequest) -> threading.Thread:
    thread = threading.Thread(target=orchestrator.free, args=(request,))
    thread.start()
    return thread


def test_model_form_frees_services_and_persistent_models_on_its_cards() -> None:
    stt = FakeService("stt", [Holder("stt", "gpu-worker", frozenset({B}))])
    tts = FakeService("tts", [Holder("tts", "qwen", frozenset({C}))])
    ls = server(model("main-old", A, B), model("describer", B, persistent=True), model("embed", C, persistent=True))
    result = make(stt, tts, ls).free(FreeRequest(reason="t", model="big"))
    assert outcomes(result) == [("stt", "gpu-worker", "released"), ("ls", "describer", "released")]
    # The server swaps its other models itself; the card the model does not use stays as it was.
    assert tts.released == [] and "main-old" not in ls.released and "embed" not in ls.released
    assert result["free"] is True
    assert [g["index"] for g in result["gpus"]] == [0, 1]


def test_model_that_evicts_nothing_leaves_persistent_models() -> None:
    ls = server(model("describer", B, persistent=True), no_eviction=("small-vlm",))
    stt = FakeService("stt", [Holder("stt", "gpu-worker", frozenset({B}))])
    result = make(stt, ls).free(FreeRequest(reason="t", model="small-vlm"))
    assert outcomes(result) == [("stt", "gpu-worker", "released")]
    assert ls.released == []


def test_already_running_skips_the_queue() -> None:
    slow = FakeService("stt", [Holder("stt", "gpu-worker", frozenset({A}))], release_delay_s=1.0)
    orchestrator = make(slow, server(model("big", A, B)))
    blocker = in_background(orchestrator, FreeRequest(reason="block", gpus=(0,), keep=("ls",)))
    time.sleep(0.2)
    started = time.monotonic()
    result = orchestrator.free(FreeRequest(reason="hook", model="big"))
    assert result["already_running"] is True and result["actions"] == []
    assert time.monotonic() - started < 0.5
    blocker.join()


def test_gpus_form_frees_main_models_too_unless_kept() -> None:
    ls = server(model("main", A), model("other", B))
    result = make(ls).free(FreeRequest(reason="t", gpus=(0, "GPU-bbbb"), keep=("ls:other",)))
    assert outcomes(result) == [("ls", "main", "released"), ("ls", "other", "kept")]
    assert result["free"] is True


def test_all_gpus() -> None:
    stt = FakeService("stt", [Holder("stt", "gpu-worker", frozenset({C}))])
    result = make(stt, server()).free(FreeRequest(reason="t", gpus="all"))
    assert outcomes(result) == [("stt", "gpu-worker", "released")]
    assert len(result["status"]["gpus"]) == 3


def test_dry_run_touches_nothing() -> None:
    stt = FakeService("stt", [Holder("stt", "gpu-worker", frozenset({A}))])
    result = make(stt, server()).free(FreeRequest(reason="t", gpus=(0,), dry_run=True))
    assert outcomes(result) == [("stt", "gpu-worker", "would_release")]
    assert stt.released == [] and result["free"] is False


def test_consumers_are_asked_in_parallel() -> None:
    slow_a = FakeService("a", [Holder("a", "x", frozenset({A}))], release_delay_s=0.5)
    slow_b = FakeService("b", [Holder("b", "y", frozenset({A}))], release_delay_s=0.5)
    started = time.monotonic()
    make(slow_a, slow_b, server()).free(FreeRequest(reason="t", gpus=(0,)))
    assert time.monotonic() - started < 0.9


def test_queue_wait_counts_against_the_overall_limit() -> None:
    slow = FakeService("stt", [Holder("stt", "gpu-worker", frozenset({A}))], release_delay_s=1.0)
    other = FakeService("tts", [Holder("tts", "qwen", frozenset({B}))])
    orchestrator = make(slow, other, server(), timeout=0.5)
    blocker = in_background(orchestrator, FreeRequest(reason="block", gpus=(0,)))
    time.sleep(0.1)
    result = orchestrator.free(FreeRequest(reason="late", gpus=(1,)))
    blocker.join()
    assert outcomes(result) == [("tts", "qwen", "timeout")]
    assert other.released == []


def test_consumer_that_cannot_report_is_an_error_and_not_free() -> None:
    result = make(FakeService("stt", fail=True), server()).free(FreeRequest(reason="t", gpus=(0,)))
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
        make(FakeService("stt"), server()).free(request_)


def test_model_server_required_when_several() -> None:
    with pytest.raises(RequestError, match="model_server"):
        make(server(), FakeServer("other")).free(FreeRequest(reason="t", model="big"))


def test_unreachable_model_server() -> None:
    with pytest.raises(ModelServerUnreachable):
        make(server(unreachable=True)).free(FreeRequest(reason="t", model="big"))


def test_status_attributes_memory_only_where_reported() -> None:
    tts = FakeService("tts", [Holder("tts", "qwen", frozenset({A}), memory_mib={A: 600})])
    status = make(tts, server(model("main", B))).status()
    by_index = {g["index"]: g for g in status["gpus"]}
    assert by_index[0]["unattributed_mib"] == 400      # 1000 used, 600 reported
    assert by_index[1]["unattributed_mib"] is None     # llama-swap reports no memory
    assert by_index[2]["unattributed_mib"] == 0 and by_index[2]["holders"] == []
    assert status["queue"] == {"running": False, "waiting": 0, "running_since_s": None}
