from typing import Any

import pytest
from fakes import GPUS
from test_orchestrator import FAST, FakeReporter, Machine, model, server, service

from gpu_orc.claims import PlaceSettings
from gpu_orc.model import Consumer, Holder
from gpu_orc.orchestrator import FreeRequest, NotFound, Orchestrator, PlaceRequest, RequestError
from gpu_orc.verify import VerifySettings

A, B, C = (gpu.uuid for gpu in GPUS)
SETTINGS = PlaceSettings(headroom_mib=512, reserve_s=60.0, priority=("tts", "stt"))


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make(*consumers: Consumer, machine: Machine | None = None, clock: Clock | None = None) -> Orchestrator:
    query = machine.query if machine is not None else lambda: list(GPUS)
    return Orchestrator(consumers, 30, query=query, verify_settings=FAST, place_settings=SETTINGS,
                        clock=clock or Clock())


def suite(name: str, *held: Holder, machine: Machine | None = None) -> FakeReporter:
    return FakeReporter(name, list(held), machine)


def place(orchestrator: Orchestrator, item: str, need: int, consumer: str = "tts", **kwargs: Any) -> dict[str, Any]:
    return orchestrator.place(PlaceRequest(consumer, item, need, "reading", **kwargs))


def card(answer: dict[str, Any]) -> int | None:
    return None if answer["card"] is None else answer["card"]["index"]


def test_a_card_nobody_uses_first_then_the_most_room() -> None:
    orchestrator = make(suite("tts"), server())
    first = place(orchestrator, "qwen", 6000)
    assert (card(first), first["reserved_s"], first["why"]) == (2, 60.0, None)   # nothing on card 2
    assert card(place(orchestrator, "xtts", 3000)) == 0                         # card 2 is promised now


def test_a_model_keeps_the_claim_of_its_profile() -> None:
    ls = server(model("big", A, B), model("elsewhere", C), claims={"big": {A: 46000, B: 31000}, "elsewhere": None})
    answer = place(make(suite("tts"), ls), "xtts", 3000)
    assert card(answer) is None
    assert answer["why"] == "needs 3000 MiB and 512 MiB headroom, the most room is 3152 MiB on GPU 0"
    assert card(place(make(suite("tts"), ls), "xtts", 2600)) == 0


def test_a_promise_counts_until_it_runs_out() -> None:
    clock = Clock()
    orchestrator = make(suite("tts"), server(), clock=clock)
    assert card(place(orchestrator, "qwen", 40000)) == 2
    assert card(place(orchestrator, "xtts", 40000)) == 0
    assert card(place(orchestrator, "piper", 40000)) is None
    clock.now += 60
    assert card(place(orchestrator, "piper", 40000)) == 2


def test_a_holder_of_the_promised_item_counts_once() -> None:
    tts = suite("tts")
    orchestrator = make(tts, server(), machine=Machine({A: 49000, B: 32000, C: 1000}))
    assert card(place(orchestrator, "qwen", 6000)) == 2
    tts.held.append(Holder("tts", "qwen", frozenset({C}), memory_mib={C: 1000}, need_mib=6000))
    gpu = orchestrator.status()["gpus"][2]
    assert gpu["available_mib"] == 49152 - 6000
    assert [(r["item"], r["need_mib"], r["left_s"]) for r in gpu["reservations"]] == [("qwen", 6000, 60.0)]


def test_the_item_it_replaces_does_not_count() -> None:
    # The worker switches models: its own holder and the memory it has in use are not in the way.
    worker = Holder("stt", "gpu-worker", frozenset({C}), memory_mib={C: 45000}, need_mib=45000, busy=False)
    machine = Machine({A: 49000, B: 32000, C: 45000})
    orchestrator = make(suite("stt", worker), server(), machine=machine)
    assert card(place(orchestrator, "gpu-worker", 48000, consumer="stt")) == 2
    assert card(place(orchestrator, "diarize-worker", 3000, consumer="stt")) is None


def test_the_named_card_and_the_compute_capability() -> None:
    orchestrator = make(suite("tts"), server())
    assert card(place(orchestrator, "qwen", 6000, card=1)) == 1
    assert card(place(orchestrator, "xtts", 3000, card="GPU-aaaa")) == 0
    assert card(place(orchestrator, "diarize", 3000, min_compute_cap=7.5)) == 2
    refused = place(orchestrator, "x", 1, min_compute_cap=9.0)
    assert (card(refused), refused["why"]) == (None, "no card of this machine meets the request")
    with pytest.raises(RequestError):
        place(orchestrator, "x", 1, card=7)


def test_only_a_side_model_service_may_ask() -> None:
    with pytest.raises(NotFound, match="'ls'.*configured: \\['tts'\\]"):
        place(make(suite("tts"), server()), "big", 1, consumer="ls")
    with pytest.raises(NotFound):
        place(make(suite("tts"), server()), "x", 1, consumer="nobody")


def test_without_every_report_nobody_can_say() -> None:
    answer = place(make(suite("tts"), FakeReporter("stt", fail=True)), "qwen", 6000)
    assert (card(answer), answer["why"]) == (None, "cannot say what is free: stt (stt is down)")


def test_an_idle_side_model_of_lower_priority_gives_way() -> None:
    worker = Holder("stt", "gpu-worker", frozenset({C}), memory_mib={C: 30000}, need_mib=30000, busy=False)
    machine = Machine({A: 49000, B: 32000, C: 30000})
    stt = suite("stt", worker, machine=machine)
    ls = server(model("big", A, B), claims={"big": {A: 49000, B: 32000}})
    answer = place(make(suite("tts"), stt, ls, machine=machine), "qwen", 20000)
    assert card(answer) == 2
    assert answer["actions"] == [{"consumer": "stt", "item": "gpu-worker", "outcome": "released"}]
    assert stt.idle_asks == ["room for tts:qwen: reading"]


def test_a_busy_one_and_a_higher_priority_stay() -> None:
    busy = Holder("stt", "gpu-worker", frozenset({C}), memory_mib={C: 30000}, need_mib=30000, busy=True)
    machine = Machine({A: 49000, B: 32000, C: 30000})
    ls = server(model("big", A, B), claims={"big": {A: 49000, B: 32000}})
    answer = place(make(suite("tts"), suite("stt", busy, machine=machine), ls, machine=machine), "qwen", 20000)
    assert (card(answer), answer["actions"]) == (None, [])
    engine = Holder("tts", "qwen", frozenset({C}), memory_mib={C: 30000}, need_mib=30000, busy=False)
    answer = place(make(suite("tts", engine), suite("stt"), ls, machine=machine), "gpu-worker", 20000, consumer="stt")
    assert (card(answer), answer["actions"]) == (None, [])


def test_one_that_got_busy_meanwhile_is_not_waited_for() -> None:
    worker = Holder("stt", "gpu-worker", frozenset({C}), memory_mib={C: 30000}, need_mib=30000, busy=False)
    machine = Machine({A: 49000, B: 32000, C: 30000})
    stt = FakeReporter("stt", [worker], machine, turns_busy=True)
    ls = server(model("big", A, B), claims={"big": {A: 49000, B: 32000}})
    orchestrator = make(suite("tts"), stt, ls, machine=machine)
    answer = place(orchestrator, "qwen", 20000)
    assert (card(answer), answer["why"]) == (None, "did not let go: stt:gpu-worker (refused_busy)")
    assert orchestrator.status()["gpus"][2]["reservations"] == []


def test_the_answer_comes_within_the_release_deadline() -> None:
    # The released memory never comes back: the check ends with the deadline of the whole answer, as for /free,
    # not a whole verify limit after the release.
    worker = Holder("stt", "gpu-worker", frozenset({C}), memory_mib={C: 30000}, need_mib=30000, busy=False)
    machine = Machine({A: 49000, B: 32000, C: 30000})
    stt = FakeReporter("stt", [worker], machine, keeps_memory=True)
    ls = server(model("big", A, B), claims={"big": {A: 49000, B: 32000}})
    slow_check = VerifySettings(timeout_s=30.0, min_freed_mib=256, poll_s=0.05)
    orchestrator = Orchestrator((suite("tts"), stt, ls), 0.3, query=machine.query, verify_settings=slow_check,
                                place_settings=SETTINGS, clock=Clock())
    answer = place(orchestrator, "qwen", 20000)
    assert card(answer) is None
    assert answer["why"].endswith("MiB room after the release")
    assert answer["duration_s"] < 2


def test_free_keeps_the_models_cards_until_it_loads() -> None:
    clock = Clock()
    ls = server(claims={"big": {A: 46000, B: 31000}})
    orchestrator = make(suite("tts"), ls, clock=clock)
    assert orchestrator.free(FreeRequest(reason="load", model="big"))["free"]
    gpus = orchestrator.status()["gpus"]
    assert [[(r["item"], r["need_mib"]) for r in gpu["reservations"]] for gpu in gpus] == [
        [("big", 46000)], [("big", 31000)], []]
    assert card(place(orchestrator, "qwen", 6000, card=0)) is None
    clock.now += 60
    assert card(place(orchestrator, "qwen", 6000, card=0)) == 0


def test_a_dry_run_promises_nothing() -> None:
    orchestrator = make(suite("tts"), server())
    orchestrator.free(FreeRequest(reason="look", model="big", dry_run=True))
    assert all(gpu["reservations"] == [] for gpu in orchestrator.status()["gpus"])


def test_a_holder_without_a_need_takes_its_whole_card() -> None:
    orchestrator = make(service("old", Holder("old", "w", frozenset({C}), memory_mib={C: 100})), suite("tts"), server())
    assert card(place(orchestrator, "qwen", 6000)) == 0
