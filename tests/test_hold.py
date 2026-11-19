import json
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fakes import GPUS
from test_orchestrator import FakeService, Machine, make, model, outcomes, server, service

import gpu_orc.orchestrator as orchestrator_module
from gpu_orc.gpus import Gpu
from gpu_orc.hold import HoldConflict, HoldFileError, HoldStore
from gpu_orc.model import Holder
from gpu_orc.orchestrator import FreeRequest, Orchestrator
from gpu_orc.work import WorkStore


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def wait_for(condition: Any, seconds: float = 5.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.02)


def started(*args: Any, **kwargs: Any) -> Orchestrator:
    """An orchestrator with its hold worker running, as in ``gpu-orc serve``."""
    orchestrator = make(*args, **kwargs)
    orchestrator.start_hold_worker()
    return orchestrator


def active(orchestrator: Orchestrator) -> Any:
    """The active hold, once the hold worker made it active."""
    wait_for(lambda: orchestrator.loading()["held"] is not None)
    return orchestrator.loading()["held"]


class BlockingFree:
    """Stands in for the orchestrator's ``free``: counts the clearings and keeps each one running until ``go``."""

    def __init__(self, orchestrator: Orchestrator, monkeypatch: pytest.MonkeyPatch) -> None:
        self.entered = threading.Event()
        self.go = threading.Event()
        self.done = threading.Event()
        self.calls = 0
        original = orchestrator.free

        def free(request: FreeRequest) -> dict[str, Any]:
            self.calls += 1
            self.entered.set()
            assert self.go.wait(5)
            try:
                return original(request)
            finally:
                self.done.set()

        monkeypatch.setattr(orchestrator, "free", free)


# --- the store -------------------------------------------------------------------------------------


def test_a_hold_outlasts_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "state" / "hold.json"
    store = HoldStore(path)
    requested = store.request("game", "gaming")
    hold = store.activate(requested, {"free": True, "actions": []})
    assert hold is not None and (hold.owner, hold.state) == ("game", "active")
    assert HoldStore(path).get() == hold


def test_a_requested_hold_outlasts_a_restart_as_requested(tmp_path: Path) -> None:
    path = tmp_path / "hold.json"
    requested = HoldStore(path).request("game", "gaming")
    assert requested.state == "requested" and requested.free is None
    assert HoldStore(path).get() == requested


def test_the_store_answers_the_own_hold_and_refuses_another_owner(tmp_path: Path) -> None:
    store = HoldStore(tmp_path / "hold.json")
    first = store.request("game", "gaming")
    assert store.request("game", "another reason") is first
    with pytest.raises(HoldConflict, match="'game' .requested"):
        store.request("render", "job")
    with pytest.raises(HoldConflict):
        store.release("render", force=False)
    assert store.get() is first


def test_only_the_very_hold_cleared_for_turns_active(tmp_path: Path) -> None:
    store = HoldStore(tmp_path / "hold.json")
    cleared_for = store.request("game", "gaming")
    store.release("game", force=False)
    again = store.request("game", "gaming")  # same owner and reason, maybe the same second: still another hold
    assert store.activate(cleared_for, {"free": True, "actions": []}) is None
    assert store.get() is again and again.state == "requested"


def test_a_hold_file_of_the_old_format_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "hold.json"
    path.write_text(json.dumps({"since": "2026-11-05T09:12:00+01:00", "reason": "gaming"}), encoding="utf-8")
    with pytest.raises(HoldFileError, match="malformed"):
        HoldStore(path)


def test_a_hold_file_with_an_unknown_state_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "hold.json"
    path.write_text(json.dumps({"owner": "o", "reason": "r", "since": "s", "state": "maybe", "free": None}), encoding="utf-8")
    with pytest.raises(HoldFileError, match="malformed"):
        HoldStore(path)


def test_letting_go_removes_the_file(tmp_path: Path) -> None:
    path = tmp_path / "hold.json"
    store = HoldStore(path)
    requested = store.request("game", "game")
    assert store.release("game", force=False) is requested
    assert store.get() is None and not path.exists() and HoldStore(path).get() is None


def test_a_malformed_hold_file_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "hold.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(HoldFileError, match="malformed"):
        HoldStore(path)


# --- requested, then active --------------------------------------------------------------------------


def test_a_hold_is_requested_first_and_the_worker_clears_every_card() -> None:
    m = Machine()
    stt = service("stt", Holder("stt", "gpu-worker", frozenset({GPUS[1].uuid})), machine=m)
    llama = server(model("big", GPUS[0].uuid), machine=m)
    orchestrator = started(stt, llama, machine=m)
    requested = orchestrator.hold("game", "gaming")["hold"]
    assert (requested["owner"], requested["reason"], requested["state"]) == ("game", "gaming", "requested")
    hold = active(orchestrator)
    assert (hold["owner"], hold["since"]) == ("game", requested["since"])
    assert outcomes(hold["free"]) == [("stt", "gpu-worker", "released"), ("ls", "big", "released")]
    light = orchestrator.loading()
    assert light["held"] == hold and light["hold_requested"] is None
    assert orchestrator.status()["held"] == hold


def test_holding_with_work_is_only_requested_and_clears_nothing() -> None:
    m = Machine()
    llama = server(model("big", GPUS[0].uuid), machine=m)
    work = WorkStore()
    work.begin("aifred", "2 running turn(s)", 60)
    orchestrator = started(llama, machine=m, work=work)
    hold = orchestrator.hold("game", "gaming")["hold"]
    time.sleep(0.2)
    assert hold["state"] == "requested" and hold["free"] is None
    assert llama.holders(list(GPUS)) != []  # still loaded: not touched
    light = orchestrator.loading()
    assert light["held"] is None and light["hold_requested"] == hold
    assert orchestrator.status()["hold_requested"] == hold


def test_the_requested_hold_activates_when_the_last_work_ends() -> None:
    m = Machine()
    llama = server(model("big", GPUS[0].uuid), machine=m)
    work = WorkStore()
    work.begin("aifred", "turn", 60)
    work.begin("job", "digest", 60)
    orchestrator = started(llama, machine=m, work=work)
    orchestrator.hold("game", "gaming")
    orchestrator.end_work("aifred")
    time.sleep(0.2)
    assert orchestrator.loading()["held"] is None  # one lease is left
    orchestrator.end_work("job")
    held = active(orchestrator)
    assert outcomes(held["free"]) == [("ls", "big", "released")]
    assert orchestrator.loading()["hold_requested"] is None


def test_the_requested_hold_activates_when_a_lease_expires() -> None:
    llama = server(model("big", GPUS[0].uuid))
    work = WorkStore()
    work.begin("crashed", "turn", 0.4)  # nobody ends it
    orchestrator = started(llama, work=work)
    orchestrator.hold("game", "gaming")
    assert orchestrator.loading()["held"] is None
    active(orchestrator)


def test_reading_the_light_never_clears_the_cards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "hold.json"
    HoldStore(path).request("game", "gaming")  # no work is reported: due, but only the hold worker may act on it
    orchestrator = make(server(model("big", GPUS[0].uuid)), hold_file=path)
    freed: list[FreeRequest] = []
    original = orchestrator.free

    def recording_free(request: FreeRequest) -> dict[str, Any]:
        freed.append(request)
        return original(request)

    monkeypatch.setattr(orchestrator, "free", recording_free)
    for _ in range(3):
        orchestrator.loading()
        orchestrator.status()
    assert freed == [] and orchestrator.loading()["hold_requested"] is not None


def test_a_requested_hold_found_at_start_waits_for_the_renewals(tmp_path: Path) -> None:
    path = tmp_path / "hold.json"
    HoldStore(path).request("game", "gaming")  # requested before the restart
    llama = server(model("big", GPUS[0].uuid))
    orchestrator = started(llama, hold_file=path, restart_grace_s=0.6)
    time.sleep(0.2)
    assert orchestrator.loading()["held"] is None and orchestrator.loading()["hold_requested"] is not None
    active(orchestrator)


def test_a_requested_hold_found_at_start_stays_requested_while_work_is_renewed(tmp_path: Path) -> None:
    path = tmp_path / "hold.json"
    HoldStore(path).request("game", "gaming")
    work = WorkStore()
    orchestrator = started(server(), hold_file=path, restart_grace_s=0.3, work=work)
    work.begin("aifred", "turn", 60)  # the renewal came back within the grace
    time.sleep(0.6)
    assert orchestrator.loading()["held"] is None


def test_the_work_of_the_holder_itself_does_not_block_the_hold() -> None:
    work = WorkStore()
    work.begin("vllm-research", "its own chain", 60)
    orchestrator = started(server(model("big", GPUS[0].uuid)), work=work)
    orchestrator.hold("vllm-research", "chain needs the cards")
    assert active(orchestrator)["owner"] == "vllm-research"


def test_the_work_of_others_blocks_even_beside_the_holders_own() -> None:
    work = WorkStore()
    work.begin("vllm-research", "own", 60)
    work.begin("aifred", "turn", 60)
    orchestrator = started(server(), work=work)
    orchestrator.hold("vllm-research", "x")
    time.sleep(0.2)
    assert orchestrator.loading()["held"] is None and orchestrator.loading()["hold_requested"] is not None


def test_a_failed_clearing_still_makes_the_hold_active_with_the_result() -> None:
    m = Machine()
    stt = service("stt", Holder("stt", "gpu-worker", frozenset({GPUS[1].uuid})), machine=m, keeps_holder=True)
    orchestrator = started(stt, machine=m)
    orchestrator.hold("game", "gaming")
    hold = active(orchestrator)
    assert hold["free"]["free"] is False and outcomes(hold["free"]) == [("stt", "gpu-worker", "not_freed")]
    assert orchestrator.status()["held"]["free"] == hold["free"]
    time.sleep(0.2)
    assert orchestrator.hold("game", "gaming")["hold"] == hold  # nobody retried


def test_the_same_owner_again_changes_nothing() -> None:
    work = WorkStore()
    work.begin("aifred", "turn", 60)
    orchestrator = started(server(), work=work)
    first = orchestrator.hold("game", "gaming")["hold"]
    assert orchestrator.hold("game", "another reason")["hold"] == first


def test_another_owner_is_refused_while_a_hold_exists_requested_or_active() -> None:
    work = WorkStore()
    work.begin("aifred", "turn", 60)
    orchestrator = started(server(), work=work)
    orchestrator.hold("game", "gaming")
    with pytest.raises(HoldConflict, match="'game' .requested"):
        orchestrator.hold("render", "job")
    orchestrator.end_work("aifred")
    active(orchestrator)
    with pytest.raises(HoldConflict, match="'game' .active"):
        orchestrator.hold("render", "job")


def test_only_the_owner_lets_go() -> None:
    orchestrator = started(server())
    orchestrator.hold("game", "gaming")
    active(orchestrator)
    with pytest.raises(HoldConflict):
        orchestrator.release_hold("render", force=False)
    assert orchestrator.loading()["held"] is not None
    assert orchestrator.release_hold("game", force=False) == {"hold": None}
    assert orchestrator.loading() == {"loading": [], "held": None, "hold_requested": None}


def test_the_owner_withdraws_a_requested_hold() -> None:
    work = WorkStore()
    work.begin("aifred", "turn", 60)
    orchestrator = started(server(), work=work)
    orchestrator.hold("game", "gaming")
    orchestrator.release_hold("game", force=False)
    orchestrator.end_work("aifred")
    time.sleep(0.2)
    assert orchestrator.loading() == {"loading": [], "held": None, "hold_requested": None}


def test_a_forced_release_lets_go_and_is_logged_with_both_owners(capsys: pytest.CaptureFixture[str]) -> None:
    orchestrator = started(server())
    orchestrator.hold("game", "gaming")
    active(orchestrator)
    orchestrator.release_hold("person", force=True)
    assert orchestrator.loading()["held"] is None
    log = capsys.readouterr().err
    assert "FORCED RELEASE" in log and "'person'" in log and "'game'" in log


# --- the hold worker: one thread for every change, nothing in between (review 19.11.2026) ------------


def test_a_hold_released_while_its_cards_are_cleared_hands_nothing_to_the_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding 1: B asked while A's clearing still ran; A's result must not make B active."""
    work = WorkStore()
    orchestrator = make(server(model("big", GPUS[0].uuid)), work=work)
    clearing = BlockingFree(orchestrator, monkeypatch)
    orchestrator.start_hold_worker()
    # In a thread: whichever thread clears, the test must not wait on it here.
    threading.Thread(target=orchestrator.hold, args=("game", "gaming"), daemon=True).start()
    assert clearing.entered.wait(5)
    orchestrator.release_hold("game", force=False)
    work.begin("aifred", "turn", 60)  # loads after the release; blocks the next hold
    orchestrator.hold("render", "job")
    clearing.go.set()
    assert clearing.done.wait(5)
    time.sleep(0.2)
    light = orchestrator.loading()
    assert light["held"] is None and light["hold_requested"]["owner"] == "render"
    assert clearing.calls == 1


def hold_workers() -> set[threading.Thread]:
    return {thread for thread in threading.enumerate() if thread.name == "hold-worker"}


def test_one_worker_lives_through_every_hold() -> None:
    """Finding 2: a thread that lived only while a hold was requested could still be ending when the
    next hold came, which then had nobody to make it active."""
    work = WorkStore()
    orchestrator = make(server(), work=work)
    before = hold_workers()
    orchestrator.start_hold_worker()
    (worker,) = hold_workers() - before
    for _ in range(3):
        work.begin("aifred", "turn", 60)
        orchestrator.hold("game", "gaming")
        orchestrator.end_work("aifred")
        active(orchestrator)
        orchestrator.release_hold("game", force=False)
    assert worker.is_alive() and hold_workers() - before == {worker}
    with pytest.raises(RuntimeError, match="runs already"):
        orchestrator.start_hold_worker()


def test_the_worker_does_not_spin_while_the_cards_are_cleared(monkeypatch: pytest.MonkeyPatch) -> None:
    """Finding 3: while a clearing ran in another thread, a woken timer looked again and again without sleeping."""
    clock = Clock()
    work = WorkStore(clock)
    work.begin("crashed", "turn", 60)  # by the fake clock; nobody ends it
    looks: list[str] = []
    original = work.blocking_for_s

    def counting_blocking_for_s(excluding_owner: str) -> float | None:
        looks.append(threading.current_thread().name)
        return original(excluding_owner)

    monkeypatch.setattr(work, "blocking_for_s", counting_blocking_for_s)
    orchestrator = make(server(model("big", GPUS[0].uuid)), work=work)
    clearing = BlockingFree(orchestrator, monkeypatch)
    orchestrator.start_hold_worker()
    orchestrator.hold("game", "gaming")  # requested: whoever waits for the lease sleeps 60 s
    wait_for(lambda: any(name != "MainThread" for name in looks))  # the background thread looked and sleeps
    time.sleep(0.05)
    clock.now += 61  # the lease has run out, nobody was told
    threading.Thread(target=orchestrator.hold, args=("game", "gaming"), daemon=True).start()
    assert clearing.entered.wait(5)
    orchestrator.end_work("nobody")  # wakes whoever sleeps
    looks.clear()
    time.sleep(0.3)
    clearing.go.set()
    assert len(looks) < 5, f"{len(looks)} looks at the work while the cards were cleared"
    assert active(orchestrator)["owner"] == "game" and clearing.calls == 1


@dataclass
class Garbled(FakeService):
    """A service whose answer has an unexpected form: reading it raises what no consumer error covers."""

    def holders(self, gpus: Sequence[Gpu]) -> list[Holder]:
        raise KeyError("item")


def test_a_clearing_that_raises_makes_the_hold_active_with_the_error(capsys: pytest.CaptureFixture[str]) -> None:
    """Finding 4: anything ``free`` raises must not end the worker and leave the hold requested unseen."""
    work = WorkStore()
    work.begin("aifred", "turn", 60)
    orchestrator = started(Garbled("stt"), server(), work=work)
    orchestrator.hold("game", "gaming")
    orchestrator.end_work("aifred")
    hold = active(orchestrator)
    assert hold["free"]["free"] is False
    assert outcomes(hold["free"]) == [("gpu-orc", "all cards", "error")]
    assert "KeyError('item')" in hold["free"]["actions"][0]["detail"]
    assert "Traceback" in capsys.readouterr().err
    orchestrator.release_hold("game", force=False)
    orchestrator.hold("game", "again")  # the worker lives on
    wait_for(lambda: orchestrator.loading()["held"] is not None)


def test_a_worker_that_fails_outside_the_clearing_ends_the_service_loudly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Review 19.11.2026: the only thread that activates holds must not die silently (every hold would stay
    requested for ever while /health is green); systemd restarts the service (Restart=on-failure)."""
    ended: list[int] = []
    ended_event = threading.Event()

    def end_process(code: int) -> None:
        ended.append(code)
        ended_event.set()

    def full_disk(self: HoldStore, expected: Any, free: dict[str, Any]) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(orchestrator_module, "_end_process", end_process)
    monkeypatch.setattr(HoldStore, "activate", full_disk)
    orchestrator = started(server())
    orchestrator.hold("game", "gaming")
    assert ended_event.wait(5)
    assert ended == [1]
    log = capsys.readouterr().err
    assert "hold worker failed" in log and "No space left on device" in log
