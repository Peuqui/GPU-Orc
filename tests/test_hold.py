import json
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fakes import GPUS
from test_orchestrator import FakeServer, FakeService, Machine, make, model, outcomes, server, service

import gpu_orc.orchestrator as orchestrator_module
from gpu_orc.gpus import Gpu
from gpu_orc.hold import HoldConflict, HoldFileError, HoldStore
from gpu_orc.model import Holder
from gpu_orc.orchestrator import FreeRequest, NotFound, Orchestrator, RequestError
from gpu_orc.work import WorkStore

FREE_RESULT: dict[str, Any] = {"free": True, "actions": []}


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


def ask(orchestrator: Orchestrator, owner: str = "game", model_name: str | None = None) -> dict[str, Any]:
    return orchestrator.hold(owner, "gaming", model_name, None)["hold"]  # type: ignore[no-any-return]


def active(orchestrator: Orchestrator) -> Any:
    """The active hold, once the hold worker made it active."""
    wait_for(lambda: orchestrator.loading()["held"] is not None)
    return orchestrator.loading()["held"]


def record_calls(monkeypatch: pytest.MonkeyPatch, target: Any, name: str) -> list[Any]:
    """Wrap ``target.name``: the first argument of each call is recorded, the call goes through."""
    calls: list[Any] = []
    original = getattr(target, name)

    def recording(argument: Any) -> Any:
        calls.append(argument)
        return original(argument)

    monkeypatch.setattr(target, name, recording)
    return calls


def stays_requested(orchestrator: Orchestrator, seconds: float = 0.4) -> Any:
    """The requested hold, after the worker had several looks (the test orchestrators check every 0.1 s)."""
    time.sleep(seconds)
    light = orchestrator.loading()
    assert light["held"] is None
    return light["hold_requested"]


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
    requested = store.request("game", "gaming", "big", "ls")
    hold = store.activate(requested, FREE_RESULT)
    assert hold is not None and (hold.owner, hold.state, hold.model, hold.model_server) == ("game", "active", "big", "ls")
    assert HoldStore(path).get() == hold


def test_a_requested_hold_outlasts_a_restart_as_requested(tmp_path: Path) -> None:
    path = tmp_path / "hold.json"
    requested = HoldStore(path).request("game", "gaming", None, None)
    assert requested.state == "requested" and requested.free is None and requested.granted_by is None
    assert requested.end_requested_by is None
    assert HoldStore(path).get() == requested


def test_the_store_answers_the_own_hold_even_with_another_model_and_refuses_another_owner(tmp_path: Path) -> None:
    store = HoldStore(tmp_path / "hold.json")
    first = store.request("game", "gaming", None, None)
    assert store.request("game", "another reason", "big", "ls") is first
    with pytest.raises(HoldConflict, match="'game' .requested"):
        store.request("render", "job", None, None)
    with pytest.raises(HoldConflict):
        store.release("render", force=False)
    assert store.get() is first


def test_only_the_very_hold_cleared_for_turns_active(tmp_path: Path) -> None:
    store = HoldStore(tmp_path / "hold.json")
    cleared_for = store.request("game", "gaming", None, None)
    store.release("game", force=False)
    again = store.request("game", "gaming", None, None)  # same owner and reason, maybe the same second: another hold
    assert store.activate(cleared_for, FREE_RESULT) is None
    assert store.get() is again and again.state == "requested"


def test_a_hold_file_without_a_model_is_an_error(tmp_path: Path) -> None:
    """The format of 0.10.0 (no model, no model_server, no granted_by, no end_requested_by): no migration."""
    path = tmp_path / "hold.json"
    path.write_text(json.dumps({"owner": "o", "reason": "r", "since": "s", "state": "requested", "free": None}),
                    encoding="utf-8")
    with pytest.raises(HoldFileError, match="malformed"):
        HoldStore(path)


def test_a_hold_file_without_granted_by_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "hold.json"
    path.write_text(json.dumps({"owner": "o", "reason": "r", "model": None, "model_server": None, "since": "s",
                                "state": "requested", "free": None}), encoding="utf-8")
    with pytest.raises(HoldFileError, match="malformed"):
        HoldStore(path)


def test_a_hold_file_without_end_requested_by_is_an_error(tmp_path: Path) -> None:
    """The format of 0.11.0: no migration."""
    path = tmp_path / "hold.json"
    path.write_text(json.dumps({"owner": "o", "reason": "r", "model": None, "model_server": None, "since": "s",
                                "state": "active", "free": FREE_RESULT, "granted_by": None}), encoding="utf-8")
    with pytest.raises(HoldFileError, match="malformed"):
        HoldStore(path)


def test_a_hold_file_with_an_end_requested_by_that_is_no_name_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "hold.json"
    path.write_text(json.dumps({"owner": "o", "reason": "r", "model": None, "model_server": None, "since": "s",
                                "state": "active", "free": FREE_RESULT, "granted_by": None, "end_requested_by": 3}),
                    encoding="utf-8")
    with pytest.raises(HoldFileError, match="malformed"):
        HoldStore(path)


def test_a_hold_file_of_the_oldest_format_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "hold.json"
    path.write_text(json.dumps({"since": "2026-11-05T09:12:00+01:00", "reason": "gaming"}), encoding="utf-8")
    with pytest.raises(HoldFileError, match="malformed"):
        HoldStore(path)


def test_a_hold_file_with_an_unknown_state_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "hold.json"
    path.write_text(json.dumps({"owner": "o", "reason": "r", "model": None, "model_server": None, "since": "s",
                                "state": "maybe", "free": None, "granted_by": None,
                                "end_requested_by": None}), encoding="utf-8")
    with pytest.raises(HoldFileError, match="malformed"):
        HoldStore(path)


def test_letting_go_removes_the_file(tmp_path: Path) -> None:
    path = tmp_path / "hold.json"
    store = HoldStore(path)
    requested = store.request("game", "game", None, None)
    assert store.release("game", force=False) is requested
    assert store.get() is None and not path.exists() and HoldStore(path).get() is None


def test_a_malformed_hold_file_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "hold.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(HoldFileError, match="malformed"):
        HoldStore(path)


def test_a_grant_is_kept_in_the_file_and_outlasts_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "hold.json"
    store = HoldStore(path)
    requested = store.request("game", "gaming", None, None)
    granted = store.grant("game", "peuqui")
    assert granted is not None and granted is not requested and granted.granted_by == "peuqui"
    assert store.get() is granted and HoldStore(path).get() == granted


def test_a_grant_replaces_the_hold_so_the_old_object_cannot_activate(tmp_path: Path) -> None:
    store = HoldStore(tmp_path / "hold.json")
    requested = store.request("game", "gaming", None, None)
    granted = store.grant("game", "peuqui")
    assert store.activate(requested, FREE_RESULT) is None
    assert granted is not None and store.activate(granted, FREE_RESULT) is not None


def test_a_grant_needs_the_owner_and_does_not_touch_an_active_hold(tmp_path: Path) -> None:
    store = HoldStore(tmp_path / "hold.json")
    assert store.grant("game", "peuqui") is None
    requested = store.request("game", "gaming", None, None)
    with pytest.raises(HoldConflict):
        store.grant("render", "peuqui")
    assert store.get() is requested
    active_hold = store.activate(requested, FREE_RESULT)
    assert store.grant("game", "peuqui") is active_hold and active_hold is not None and active_hold.granted_by is None


def test_a_grant_does_not_carry_over_to_a_hold_requested_anew(tmp_path: Path) -> None:
    store = HoldStore(tmp_path / "hold.json")
    store.request("game", "gaming", None, None)
    store.grant("game", "peuqui")
    store.release("game", force=False)
    assert store.request("game", "gaming", None, None).granted_by is None


def test_an_end_request_of_an_active_hold_is_kept_in_the_file_and_outlasts_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "hold.json"
    store = HoldStore(path)
    active_hold = store.activate(store.request("game", "gaming", None, None), FREE_RESULT)
    marked = store.request_end("game", "peuqui")
    assert active_hold is not None and marked is not None and marked is not active_hold
    assert (marked.state, marked.end_requested_by) == ("active", "peuqui")
    assert store.get() is marked and HoldStore(path).get() == marked


def test_a_second_end_request_changes_nothing(tmp_path: Path) -> None:
    store = HoldStore(tmp_path / "hold.json")
    store.activate(store.request("game", "gaming", None, None), FREE_RESULT)
    first = store.request_end("game", "peuqui")
    assert store.request_end("game", "somebody else") is first
    assert first is not None and first.end_requested_by == "peuqui"


def test_an_end_request_withdraws_a_requested_hold(tmp_path: Path) -> None:
    path = tmp_path / "hold.json"
    store = HoldStore(path)
    requested = store.request("game", "gaming", None, None)
    assert store.request_end("game", "peuqui") is requested
    assert store.get() is None and not path.exists() and HoldStore(path).get() is None
    assert store.activate(requested, FREE_RESULT) is None  # the worker's clearing counts for nothing


def test_an_end_request_needs_a_hold_and_its_owner(tmp_path: Path) -> None:
    store = HoldStore(tmp_path / "hold.json")
    assert store.request_end("game", "peuqui") is None
    requested = store.request("game", "gaming", None, None)
    with pytest.raises(HoldConflict, match="'game' .requested"):
        store.request_end("render", "peuqui")
    assert store.get() is requested


def test_an_end_request_does_not_carry_over_to_a_hold_requested_anew(tmp_path: Path) -> None:
    store = HoldStore(tmp_path / "hold.json")
    store.activate(store.request("game", "gaming", None, None), FREE_RESULT)
    store.request_end("game", "peuqui")
    store.release("game", force=False)
    assert store.request("game", "gaming", None, None).end_requested_by is None


# --- requested, then active --------------------------------------------------------------------------


def test_free_cards_make_a_new_hold_active_at_once_and_everything_is_cleared() -> None:
    m = Machine()
    stt = service("stt", Holder("stt", "gpu-worker", frozenset({GPUS[1].uuid})), machine=m)
    llama = server(model("embed", GPUS[2].uuid, persistent=True), machine=m)  # a helper does not count
    orchestrator = started(stt, llama, machine=m)
    requested = ask(orchestrator)
    assert (requested["owner"], requested["reason"], requested["state"]) == ("game", "gaming", "requested")
    assert (requested["model"], requested["model_server"], requested["granted_by"]) == (None, None, None)
    hold = active(orchestrator)
    assert (hold["owner"], hold["since"]) == ("game", requested["since"])
    assert outcomes(hold["free"]) == [("stt", "gpu-worker", "released"), ("ls", "embed", "released")]
    light = orchestrator.loading()
    assert light["held"] == hold and light["hold_requested"] is None
    assert orchestrator.status()["held"] == hold


def test_a_loaded_main_model_keeps_the_hold_requested_and_clears_nothing() -> None:
    llama = server(model("big", GPUS[0].uuid))
    orchestrator = started(llama)
    hold = ask(orchestrator)
    assert hold["state"] == "requested" and hold["free"] is None
    assert stays_requested(orchestrator) == hold
    assert llama.holders(list(GPUS)) != [] and llama.released == []  # not touched
    assert orchestrator.status()["hold_requested"] == hold


def test_the_requested_hold_activates_after_the_interval_when_the_main_model_is_gone() -> None:
    m = Machine()
    llama = server(model("big", GPUS[0].uuid), machine=m)
    orchestrator = started(llama, machine=m)
    ask(orchestrator)
    stays_requested(orchestrator, 0.3)
    llama.held.clear()  # the model server swapped it out
    held = active(orchestrator)
    assert outcomes(held["free"]) == []
    assert orchestrator.loading()["hold_requested"] is None


def test_the_worker_looks_again_only_after_the_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    llama = server(model("big", GPUS[0].uuid))
    orchestrator = make(llama, hold_check_interval_s=30.0)
    looks = record_calls(monkeypatch, orchestrator, "_cards_free")
    orchestrator.start_hold_worker()
    ask(orchestrator)
    time.sleep(0.5)
    assert len(looks) == 1  # one look when asked, then it sleeps


def test_the_requested_model_running_counts_as_free_and_is_kept() -> None:
    m = Machine()
    stt = service("stt", Holder("stt", "gpu-worker", frozenset({GPUS[1].uuid})), machine=m)
    llama = server(model("big", GPUS[0].uuid), machine=m)
    orchestrator = started(stt, llama, machine=m)
    requested = orchestrator.hold("vllm-research", "chain", "big", None)["hold"]
    assert (requested["model"], requested["model_server"]) == ("big", "ls")
    hold = active(orchestrator)
    assert outcomes(hold["free"]) == [("stt", "gpu-worker", "released"), ("ls", "big", "kept")]
    assert llama.released == []
    assert (hold["model"], hold["model_server"]) == ("big", "ls")


def test_another_main_model_than_the_requested_one_blocks() -> None:
    llama = server(model("small-vlm", GPUS[1].uuid))
    orchestrator = started(llama)
    ask(orchestrator, model_name="big")
    assert stays_requested(orchestrator) is not None


def test_a_helper_does_not_block() -> None:
    llama = server(model("embed", GPUS[2].uuid, persistent=True))
    orchestrator = started(llama)
    ask(orchestrator)
    assert outcomes(active(orchestrator)["free"]) == [("ls", "embed", "released")]


def test_a_model_server_that_is_loading_blocks() -> None:
    llama = server(starting=("small-vlm",))
    orchestrator = started(llama)
    ask(orchestrator, model_name="big")
    assert stays_requested(orchestrator) is not None
    llama.starting = ()
    active(orchestrator)


def test_a_model_announced_by_a_free_counts_as_loading_before_the_server_says_so() -> None:
    orchestrator = started(server())  # llama-swap reports nothing
    orchestrator._announce("small-vlm", time.monotonic() + 60)
    ask(orchestrator, model_name="big")
    assert stays_requested(orchestrator) is not None
    orchestrator._announce("small-vlm", time.monotonic() - 1)  # the announcement ran out
    active(orchestrator)


def test_only_the_requested_model_announced_does_not_block() -> None:
    orchestrator = started(server())
    orchestrator._announce("big", time.monotonic() + 60)
    ask(orchestrator, model_name="big")
    active(orchestrator)


def test_the_requested_model_alone_loading_does_not_block() -> None:
    orchestrator = started(server(starting=("big",)))
    ask(orchestrator, model_name="big")
    active(orchestrator)


def test_a_model_server_that_does_not_answer_blocks_and_is_reported_once(capsys: pytest.CaptureFixture[str]) -> None:
    llama = server(unreachable=True)
    orchestrator = started(llama)
    ask(orchestrator)
    time.sleep(0.6)  # about six looks; the light itself cannot be read while the server is down
    status = orchestrator.status()
    assert status["held"] is None and status["hold_requested"] is not None
    log = capsys.readouterr().err
    assert log.count("cannot tell whether the cards are free") == 1 and "down" in log
    llama.unreachable = False
    active(orchestrator)


def test_the_other_consumers_do_not_count() -> None:
    m = Machine()
    stt = service("stt", Holder("stt", "gpu-worker", frozenset({GPUS[1].uuid})), machine=m)
    orchestrator = started(stt, server(machine=m), machine=m)
    ask(orchestrator)
    assert outcomes(active(orchestrator)["free"]) == [("stt", "gpu-worker", "released")]


def test_reported_work_does_not_influence_a_hold_without_a_grant() -> None:
    work = WorkStore()
    work.begin("aifred", "turn", 60)
    orchestrator = started(server(), work=work)
    ask(orchestrator)
    active(orchestrator)  # free cards: no click, no work condition


def test_a_requested_hold_found_at_start_is_looked_at_by_the_worker(tmp_path: Path) -> None:
    path = tmp_path / "hold.json"
    HoldStore(path).request("game", "gaming", None, None)  # requested before the restart
    active(started(server(), hold_file=path))


def test_reading_the_light_never_clears_the_cards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "hold.json"
    HoldStore(path).request("game", "gaming", None, None)  # due, but only the hold worker may act on it
    orchestrator = make(server(), hold_file=path)
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


def test_the_clearing_request_names_the_owner_the_reason_and_keeps_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = make(server(model("big", GPUS[0].uuid)))
    requests = record_calls(monkeypatch, orchestrator, "free")
    orchestrator.start_hold_worker()
    orchestrator.hold("vllm-research", "chain", "big", "ls")
    active(orchestrator)
    assert requests == [FreeRequest(reason="hold: vllm-research: chain", gpus="all", keep=("ls:big",))]


def test_a_hold_without_a_model_keeps_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = make(server())
    requests = record_calls(monkeypatch, orchestrator, "free")
    orchestrator.start_hold_worker()
    ask(orchestrator)
    active(orchestrator)
    assert requests == [FreeRequest(reason="hold: game: gaming", gpus="all", keep=())]


def test_a_failed_clearing_still_makes_the_hold_active_with_the_result() -> None:
    m = Machine()
    stt = service("stt", Holder("stt", "gpu-worker", frozenset({GPUS[1].uuid})), machine=m, keeps_holder=True)
    orchestrator = started(stt, server(), machine=m)
    ask(orchestrator)
    hold = active(orchestrator)
    assert hold["free"]["free"] is False and outcomes(hold["free"]) == [("stt", "gpu-worker", "not_freed")]
    assert orchestrator.status()["held"]["free"] == hold["free"]
    time.sleep(0.2)
    assert orchestrator.hold("game", "gaming", None, None)["hold"] == hold  # nobody retried


# --- asking: model, model server, owner ----------------------------------------------------------------


def test_a_hold_for_an_unknown_model_is_refused() -> None:
    orchestrator = started(server())
    with pytest.raises(RequestError, match="'ghost' is not in the config of ls"):
        orchestrator.hold("game", "gaming", "ghost", None)
    assert orchestrator.loading()["hold_requested"] is None


def test_a_hold_for_a_model_needs_the_model_server_with_several() -> None:
    other = FakeServer("vllm", [], None, models={})
    orchestrator = started(server(), other)
    with pytest.raises(RequestError, match="model_server is required"):
        orchestrator.hold("game", "gaming", "big", None)
    with pytest.raises(NotFound, match="no model server 'nope'"):
        orchestrator.hold("game", "gaming", "big", "nope")
    assert orchestrator.hold("game", "gaming", "big", "ls")["hold"]["model_server"] == "ls"


def test_a_hold_without_a_model_needs_no_model_server() -> None:
    orchestrator = started(FakeServer("vllm", [], None), server())  # two model servers, none named
    hold = ask(orchestrator)
    assert (hold["model"], hold["model_server"]) == (None, None)


def test_the_same_owner_again_changes_nothing_even_with_another_model() -> None:
    orchestrator = started(server(model("big", GPUS[0].uuid)))
    first = ask(orchestrator)
    assert orchestrator.hold("game", "another reason", "big", None)["hold"] == first


def test_another_owner_is_refused_while_a_hold_exists_requested_or_active() -> None:
    llama = server(model("big", GPUS[0].uuid))
    orchestrator = started(llama)
    ask(orchestrator)
    with pytest.raises(HoldConflict, match="'game' .requested"):
        ask(orchestrator, "render")
    llama.held.clear()
    active(orchestrator)
    with pytest.raises(HoldConflict, match="'game' .active"):
        ask(orchestrator, "render")


def test_only_the_owner_lets_go() -> None:
    orchestrator = started(server())
    ask(orchestrator)
    active(orchestrator)
    with pytest.raises(HoldConflict):
        orchestrator.release_hold("render", force=False)
    assert orchestrator.loading()["held"] is not None
    assert orchestrator.release_hold("game", force=False) == {"hold": None}
    assert orchestrator.loading() == {"loading": [], "held": None, "hold_requested": None}


def test_the_owner_withdraws_a_requested_hold() -> None:
    llama = server(model("big", GPUS[0].uuid))
    orchestrator = started(llama)
    ask(orchestrator)
    orchestrator.release_hold("game", force=False)
    llama.held.clear()
    time.sleep(0.3)
    assert orchestrator.loading() == {"loading": [], "held": None, "hold_requested": None}


def test_a_forced_release_lets_go_and_is_logged_with_both_owners(capsys: pytest.CaptureFixture[str]) -> None:
    orchestrator = started(server())
    ask(orchestrator)
    active(orchestrator)
    orchestrator.release_hold("person", force=True)
    assert orchestrator.loading()["held"] is None
    log = capsys.readouterr().err
    assert "FORCED RELEASE" in log and "'person'" in log and "'game'" in log


# --- the grant -------------------------------------------------------------------------------------


def test_a_grant_makes_the_hold_active_although_a_main_model_is_loaded(capsys: pytest.CaptureFixture[str]) -> None:
    m = Machine()
    llama = server(model("big", GPUS[0].uuid), machine=m)
    orchestrator = started(llama, machine=m)
    ask(orchestrator)
    assert stays_requested(orchestrator) is not None
    answer = orchestrator.grant_hold("game", "peuqui")
    assert answer["hold"]["state"] == "requested" and answer["hold"]["granted_by"] == "peuqui"
    hold = active(orchestrator)
    assert outcomes(hold["free"]) == [("ls", "big", "released")] and hold["granted_by"] == "peuqui"
    assert "[hold] GRANTED by 'peuqui' for 'game'" in capsys.readouterr().err


def test_the_light_shows_who_granted_a_requested_hold() -> None:
    work = WorkStore()
    work.begin("aifred", "turn", 60)
    orchestrator = started(server(model("big", GPUS[0].uuid)), work=work)
    ask(orchestrator)
    orchestrator.grant_hold("game", "peuqui")
    assert stays_requested(orchestrator)["granted_by"] == "peuqui"
    assert orchestrator.status()["hold_requested"]["granted_by"] == "peuqui"


def test_a_grant_waits_for_the_work_of_others_and_the_end_of_it_wakes_the_worker() -> None:
    llama = server(model("big", GPUS[0].uuid))
    work = WorkStore()
    work.begin("aifred", "an answer runs", 60)
    orchestrator = started(llama, work=work, hold_check_interval_s=30.0)  # only the wake can be quick
    ask(orchestrator)
    orchestrator.grant_hold("game", "peuqui")
    assert stays_requested(orchestrator)["granted_by"] == "peuqui"
    assert llama.released == []  # the running answer is not broken off
    orchestrator.end_work("aifred")
    assert active(orchestrator)["granted_by"] == "peuqui"
    assert llama.released == ["big"]


def test_the_work_of_the_owner_does_not_block_a_grant() -> None:
    work = WorkStore()
    work.begin("vllm-research", "its own chain", 60)
    orchestrator = started(server(model("big", GPUS[0].uuid)), work=work)
    orchestrator.hold("vllm-research", "chain", None, None)
    orchestrator.grant_hold("vllm-research", "peuqui")
    assert active(orchestrator)["owner"] == "vllm-research"


def test_the_work_of_others_blocks_a_grant_even_beside_the_owners_own() -> None:
    work = WorkStore()
    work.begin("vllm-research", "own", 60)
    work.begin("aifred", "turn", 60)
    orchestrator = started(server(model("big", GPUS[0].uuid)), work=work)
    orchestrator.hold("vllm-research", "x", None, None)
    orchestrator.grant_hold("vllm-research", "peuqui")
    assert stays_requested(orchestrator) is not None


def test_a_grant_of_another_owner_is_refused() -> None:
    orchestrator = started(server(model("big", GPUS[0].uuid)))
    ask(orchestrator)
    with pytest.raises(HoldConflict, match="'game' .requested"):
        orchestrator.grant_hold("render", "peuqui")
    assert stays_requested(orchestrator, 0.2)["granted_by"] is None


def test_a_grant_without_a_hold_is_not_found() -> None:
    with pytest.raises(NotFound, match="no hold"):
        started(server()).grant_hold("game", "peuqui")


def test_a_grant_of_an_active_hold_changes_nothing() -> None:
    orchestrator = started(server())
    ask(orchestrator)
    hold = active(orchestrator)
    assert orchestrator.grant_hold("game", "peuqui") == {"hold": hold}
    assert orchestrator.loading()["held"] == hold


def test_a_grant_does_not_hold_for_a_hold_requested_anew() -> None:
    llama = server(model("big", GPUS[0].uuid))
    orchestrator = started(llama)
    ask(orchestrator)
    orchestrator.grant_hold("game", "peuqui")
    wait_for(lambda: orchestrator.loading()["held"] is not None)
    orchestrator.release_hold("game", force=False)
    llama.held.append(model("big", GPUS[0].uuid))  # something loads again
    renewed = ask(orchestrator)
    assert renewed["granted_by"] is None
    assert stays_requested(orchestrator) == renewed


def test_a_grant_outlasts_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "hold.json"
    work = WorkStore()
    work.begin("aifred", "turn", 60)
    first = make(server(model("big", GPUS[0].uuid)), hold_file=path, work=work)
    first.hold("game", "gaming", None, None)
    first.grant_hold("game", "peuqui")  # no worker: the service goes down before the click is acted on
    llama = server(model("big", GPUS[0].uuid))
    restarted = started(llama, hold_file=path)
    assert active(restarted)["granted_by"] == "peuqui" and llama.released == ["big"]


# --- the end request ------------------------------------------------------------------------------


def test_an_end_request_marks_an_active_hold_and_keeps_the_cards_locked(capsys: pytest.CaptureFixture[str]) -> None:
    orchestrator = started(server())
    ask(orchestrator)
    hold = active(orchestrator)
    answer = orchestrator.request_hold_end("game", "peuqui")
    assert answer == {"hold": {**hold, "end_requested_by": "peuqui"}}
    assert orchestrator.loading()["held"] == answer["hold"] and orchestrator.loading()["hold_requested"] is None
    assert orchestrator.status()["held"]["end_requested_by"] == "peuqui"
    assert "[hold] END REQUESTED by 'peuqui' for 'game'" in capsys.readouterr().err


def test_an_end_request_withdraws_a_requested_hold_at_once(capsys: pytest.CaptureFixture[str]) -> None:
    orchestrator = started(server(model("big", GPUS[0].uuid)))
    ask(orchestrator)
    assert stays_requested(orchestrator) is not None
    assert orchestrator.request_hold_end("game", "peuqui") == {"hold": None}
    assert orchestrator.loading() == {"loading": [], "held": None, "hold_requested": None}
    assert "[hold] WITHDRAWN on request of 'peuqui': 'game'" in capsys.readouterr().err


def test_an_end_request_of_another_owner_is_refused() -> None:
    orchestrator = started(server())
    ask(orchestrator)
    active(orchestrator)
    with pytest.raises(HoldConflict, match="'game' .active"):
        orchestrator.request_hold_end("render", "peuqui")
    assert orchestrator.loading()["held"]["end_requested_by"] is None


def test_an_end_request_without_a_hold_is_not_found() -> None:
    with pytest.raises(NotFound, match="no hold"):
        started(server()).request_hold_end("game", "peuqui")


def test_an_end_request_outlasts_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "hold.json"
    first = started(server(), hold_file=path)
    ask(first)
    active(first)
    first.request_hold_end("game", "peuqui")
    restarted = make(server(), hold_file=path)
    assert restarted.loading()["held"]["end_requested_by"] == "peuqui"


# --- the hold worker: one thread for every change, nothing in between (review 19.11.2026) ------------


def test_a_hold_released_while_its_cards_are_cleared_hands_nothing_to_the_next(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Finding 1: B asked while A's clearing still ran; A's result must not make B active, B is cleared for itself."""
    orchestrator = make(server())
    clearing = BlockingFree(orchestrator, monkeypatch)
    orchestrator.start_hold_worker()
    # In a thread: whichever thread clears, the test must not wait on it here.
    threading.Thread(target=ask, args=(orchestrator, "game"), daemon=True).start()
    assert clearing.entered.wait(5)
    orchestrator.release_hold("game", force=False)
    ask(orchestrator, "render")
    clearing.go.set()
    hold = active(orchestrator)
    assert hold["owner"] == "render" and clearing.calls == 2  # not handed A's result: cleared again for B
    assert "was released or granted while the cards were cleared" in capsys.readouterr().err


def hold_workers() -> set[threading.Thread]:
    return {thread for thread in threading.enumerate() if thread.name == "hold-worker"}


def test_one_worker_lives_through_every_hold() -> None:
    """Finding 2: a thread that lived only while a hold was requested could still be ending when the
    next hold came, which then had nobody to make it active."""
    orchestrator = make(server())
    before = hold_workers()
    orchestrator.start_hold_worker()
    (worker,) = hold_workers() - before
    for _ in range(3):
        ask(orchestrator)
        active(orchestrator)
        orchestrator.release_hold("game", force=False)
    assert worker.is_alive() and hold_workers() - before == {worker}
    with pytest.raises(RuntimeError, match="runs already"):
        orchestrator.start_hold_worker()


@dataclass
class Garbled(FakeService):
    """A service whose answer has an unexpected form: reading it raises what no consumer error covers."""

    def holders(self, gpus: Sequence[Gpu]) -> list[Holder]:
        raise KeyError("item")


def test_a_clearing_that_raises_makes_the_hold_active_with_the_error(capsys: pytest.CaptureFixture[str]) -> None:
    """Finding 4: anything ``free`` raises must not end the worker and leave the hold requested unseen."""
    orchestrator = started(Garbled("stt"), server())
    ask(orchestrator)
    hold = active(orchestrator)
    assert hold["free"]["free"] is False
    assert outcomes(hold["free"]) == [("gpu-orc", "all cards", "error")]
    assert "KeyError('item')" in hold["free"]["actions"][0]["detail"]
    assert "Traceback" in capsys.readouterr().err
    orchestrator.release_hold("game", force=False)
    ask(orchestrator)  # the worker lives on
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
    ask(orchestrator)
    assert ended_event.wait(5)
    assert ended == [1]
    log = capsys.readouterr().err
    assert "hold worker failed" in log and "No space left on device" in log
