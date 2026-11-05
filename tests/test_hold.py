from pathlib import Path

import pytest
from fakes import GPUS
from test_orchestrator import Machine, make, model, outcomes, server, service

from gpu_orc.hold import HoldFileError, HoldStore
from gpu_orc.model import Holder


def test_a_hold_outlasts_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "state" / "hold.json"
    hold = HoldStore(path).set("gaming")
    assert HoldStore(path).get() == hold


def test_holding_again_keeps_the_time_and_takes_the_new_reason(tmp_path: Path) -> None:
    store = HoldStore(tmp_path / "hold.json")
    first = store.set("game")
    second = store.set("render job")
    assert (second.since, second.reason) == (first.since, "render job")


def test_letting_go_removes_the_file(tmp_path: Path) -> None:
    path = tmp_path / "hold.json"
    store = HoldStore(path)
    store.set("game")
    store.clear()
    assert store.get() is None and not path.exists() and HoldStore(path).get() is None


def test_a_malformed_hold_file_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "hold.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(HoldFileError, match="malformed"):
        HoldStore(path)


def test_holding_clears_every_card_and_shows_on_the_traffic_light() -> None:
    m = Machine()
    stt = service("stt", Holder("stt", "gpu-worker", frozenset({GPUS[1].uuid})), machine=m)
    llama = server(model("big", GPUS[0].uuid), machine=m)
    orchestrator = make(stt, llama, machine=m)
    answer = orchestrator.hold("game")
    assert answer["held"]["reason"] == "game"
    assert outcomes(answer["free"]) == [("stt", "gpu-worker", "released"), ("ls", "big", "released")]
    assert orchestrator.loading()["held"] == answer["held"]
    assert orchestrator.status()["held"] == answer["held"]


def test_letting_go_turns_the_light_green_again() -> None:
    orchestrator = make(server())
    orchestrator.hold("game")
    assert orchestrator.release_hold() == {"held": None}
    assert orchestrator.loading() == {"loading": [], "held": None}
