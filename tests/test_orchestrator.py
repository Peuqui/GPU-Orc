import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import pytest
from fakes import GPUS

from gpu_orc.gpus import Gpu, GpuSpecError
from gpu_orc.measurements import Measurements, VisionMeasurement
from gpu_orc.model import Action, Consumer, ConsumerError, Holder, Need, UnknownModel
from gpu_orc.orchestrator import FreeRequest, ModelServerUnreachable, NotFound, Orchestrator, RequestError
from gpu_orc.profiles import Profile, Variant
from gpu_orc.verify import VerifySettings

A, B, C = (gpu.uuid for gpu in GPUS)
FAST = VerifySettings(timeout_s=0.3, min_freed_mib=256, poll_s=0.05)
HOLDER_MIB = 1000


class Machine:
    """GPU memory that goes down when a fake service really lets go of what it held."""

    def __init__(self, used: dict[str, int] | None = None) -> None:
        self.used = used if used is not None else {gpu.uuid: gpu.memory_used_mib + 5000 for gpu in GPUS}
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
    # The claim of a profile per card; a model not named here claims 1000 MiB of each of its cards.
    claims: dict[str, dict[str, int] | None] = field(default_factory=dict)

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

    def profiles(self, gpus: Sequence[Gpu], measurements: Measurements) -> list[Profile]:
        if self.unreachable:
            raise ConsumerError("config unreadable")
        return [Profile(name, "vllm", tuple(sorted(cards)), False, 307200, 0.9, 3500, Variant(name),
                        self.claims.get(name, {uuid: 1000 for uuid in cards}), "fraction", None)
                for name, cards in self.models.items()]


@dataclass
class FakeReporter(FakeService):
    """A suite: it reports needs and lets go of idle items."""

    reported: list[Need] = field(default_factory=list)
    idle_asks: list[str] = field(default_factory=list)
    turns_busy: bool = False  # got busy between the plan and the ask

    def needs(self) -> list[Need]:
        if self.fail:
            raise ConsumerError(f"{self.name} is down")
        return list(self.reported)

    def release_idle(self, holders: Sequence[Holder], *, reason: str, deadline: float) -> list[Action]:
        self.idle_asks.append(reason)
        busy = [holder for holder in holders if holder.busy or self.turns_busy]
        actions = self.release([holder for holder in holders if holder not in busy], reason=reason, deadline=deadline)
        return actions + [Action(self.name, holder.item, "refused_busy") for holder in busy]


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


def test_a_card_that_held_nothing_yet_cannot_go_down() -> None:
    # Found live (25.10.): a TTS start withdrawn before its container had a CUDA context; the card stayed at 0 MiB.
    m = Machine()
    m.used[A] = 0
    tts = service("tts", Holder("tts", "xtts", frozenset({A})), machine=None)
    result = make(tts, server(), machine=m).free(FreeRequest(reason="t", gpus=(0,)))
    assert outcomes(result) == [("tts", "xtts", "released")] and result["free"] is True


def test_a_card_with_little_in_use_must_give_that_back() -> None:
    m = Machine()
    m.used[A] = 100
    tts = service("tts", Holder("tts", "xtts", frozenset({A})), machine=m, keeps_memory=True)
    action = make(tts, server(), machine=m).free(FreeRequest(reason="t", gpus=(0,)))["actions"][0]
    assert action["outcome"] == "not_freed" and "−0 MiB over 1 card(s), expected at least −100" in action["detail"]


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
    tts = service("tts", Holder("tts", "qwen", frozenset({A}), memory_mib={A: 600}, need_mib=600))
    status = make(tts, server(model("big", B))).status()
    by_index = {g["index"]: g for g in status["gpus"]}
    assert by_index[0]["unattributed_mib"] == 400      # 1000 used, 600 reported
    assert by_index[1]["unattributed_mib"] is None     # llama-swap reports no memory
    assert by_index[2]["unattributed_mib"] == 0 and by_index[2]["holders"] == []
    assert status["queue"] == {"running": False, "waiting": 0, "running_since_s": None}


def test_available_leaves_every_holder_its_claim() -> None:
    main = model("big", A, B)
    machine = Machine({A: 20000, B: 6000, C: 2000})
    tts = service("tts", Holder("tts", "qwen", frozenset({C}), memory_mib={C: 2000}, need_mib=6000))
    stt = service("stt", Holder("stt", "gpu-worker", frozenset({A}), memory_mib={A: 3000}, need_mib=1030))
    ls = server(main, claims={"big": {A: 40000, B: 30000}})
    by_index = {g["index"]: g for g in make(tts, stt, ls, machine=machine).status()["gpus"]}
    # The model may fill its claim while it computes; the worker holds more than it needs; the engine still loads.
    assert by_index[0]["available_mib"] == 49152 - 40000 - 3000
    assert by_index[1]["available_mib"] == 32768 - 30000
    assert by_index[2]["available_mib"] == 49152 - 6000
    assert [(h["item"], h["need_mib"], h["claim_mib"]) for h in by_index[0]["holders"]] == [
        ("gpu-worker", 1030, 3000), ("big", None, 40000)]


def test_memory_no_holder_accounts_for_counts_on_top() -> None:
    tts = service("tts", Holder("tts", "qwen", frozenset({C}), memory_mib={C: 2000}, need_mib=6000))
    status = make(tts, server(), machine=Machine({A: 0, B: 0, C: 9000})).status()
    assert status["gpus"][2]["available_mib"] == 49152 - 6000 - 7000


def test_an_unknown_claim_takes_the_whole_card() -> None:
    tts = service("tts", Holder("tts", "qwen", frozenset({C}), memory_mib={C: 2000}))
    ls = server(model("big", A), claims={"big": None})
    gpus = make(tts, ls).status()["gpus"]
    assert [g["available_mib"] for g in gpus] == [0, 32768 - 500, 0]


def test_without_every_consumer_nobody_can_say_what_is_available() -> None:
    status = make(service("tts", fail=True), server()).status()
    assert [g["available_mib"] for g in status["gpus"]] == [None, None, None]


def test_needs_are_the_suites_reports_and_the_measured_vision_models() -> None:
    tts = FakeReporter("tts", reported=[Need("tts", "xtts", 3680, "estimate")])
    measured = Measurements(vision={"vl-4b": VisionMeasurement(7396, 24576, "2026-10-09", "x")})
    orchestrator = Orchestrator([tts, server()], 30, query=lambda: list(GPUS), verify_settings=FAST,
                                measurements=lambda: measured, vision_models={"ls": ["vl-4b", "vl-new"]})
    assert orchestrator.needs() == {"needs": [
        {"consumer": "tts", "item": "xtts", "need_mib": 3680, "source": "estimate", "default": False},
        {"consumer": "ls", "item": "vl-4b", "need_mib": 7396, "source": "measured", "default": False},
        {"consumer": "ls", "item": "vl-new", "need_mib": None, "source": "unknown", "default": False},
    ], "errors": []}


def test_a_suite_that_cannot_report_its_needs_is_named() -> None:
    xtts = Need("tts", "xtts", 3680, "estimate")
    answer = make(FakeReporter("stt", fail=True), FakeReporter("tts", reported=[xtts])).needs()
    assert [need["item"] for need in answer["needs"]] == ["xtts"]
    assert answer["errors"] == [{"consumer": "stt", "detail": "stt is down"}]


def test_profiles_come_from_the_model_servers() -> None:
    answer = make(service("stt"), server()).profiles()
    assert answer["errors"] == []
    big = next(profile for profile in answer["profiles"] if profile["name"] == "big")
    assert big["model_server"] == "ls" and big["cards"] == sorted([A, B]) and big["claim_source"] == "fraction"
    assert big["variant"] == {"base": "big", "tts": None, "vlm": None, "speed": False, "describer": False,
                              "describer_gpu": None}
    assert make(server(unreachable=True)).profiles() == {"profiles": [], "errors": [
        {"consumer": "ls", "detail": "config unreadable"}]}
