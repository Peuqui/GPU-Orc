import time
from typing import Any

import pytest
from fakes import GPUS, FakeService, answers

from gpu_orc.consumers import tts_suite
from gpu_orc.consumers.tts_suite import TtsSuite
from gpu_orc.model import Holder

ALL = frozenset(gpu.uuid for gpu in GPUS)


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tts_suite, "POLL_S", 0.05)


def engine(name: str, status: str = "running", host: str | None = None, needs_gpu: bool = True,
           vram: dict[str, int] | None = None) -> dict[str, Any]:
    return {"engine": name, "status": status, "host": host, "needs_gpu": needs_gpu, "vram_mib": vram}


def test_holders_are_local_gpu_engines_that_run() -> None:
    engines = [
        engine("qwen", vram={"GPU-cccc2222": 6024}),
        engine("xtts", status="starting", vram={}),
        engine("remote", host="other-host", vram={"GPU-x": 1}),
        engine("edge", needs_gpu=False),
        engine("idle", status="stopped", vram={}),
        engine("unmeasured", vram=None),
    ]
    with FakeService({("GET", "/engines"): answers((200, engines))}).running() as fake:
        holders = TtsSuite("tts", fake.url).holders(GPUS)
    assert [(h.item, h.gpu_uuids, dict(h.memory_mib)) for h in holders] == [
        ("qwen", frozenset({"GPU-cccc2222"}), {"GPU-cccc2222": 6024}),
        ("xtts", frozenset(), {}),
        # Unmeasured: it may sit on any card.
        ("unmeasured", ALL, {}),
    ]


def held(*names: str) -> list[Holder]:
    return [Holder("tts", name, frozenset({"GPU-cccc2222"})) for name in names]


def results(*pairs: tuple[str, str]) -> list[dict[str, Any]]:
    return [{"engine": name, "host": None, "outcome": outcome, "detail": ""} for name, outcome in pairs]


def test_release_maps_the_service_outcomes() -> None:
    reply = results(("a", "stopped"), ("b", "not_running"), ("c", "failed"))
    with FakeService({("POST", "/release"): answers((200, reply))}).running() as fake:
        actions = TtsSuite("tts", fake.url).release(held("a", "b", "c"), reason="r", deadline=time.monotonic() + 60)
        assert fake.calls[0][2] == {"targets": [{"engine": n, "host": None} for n in "abc"]}
    assert [(a.item, a.outcome) for a in actions] == [("a", "released"), ("b", "released"), ("c", "error")]


def test_reading_in_progress_is_refused_without_patience() -> None:
    with FakeService({("POST", "/release"): answers((200, results(("a", "in_use"))))}).running() as fake:
        actions = TtsSuite("tts", fake.url).release(held("a"), reason="r", deadline=time.monotonic() + 60)
        assert len(fake.calls) == 1
    assert actions[0].outcome == "refused_busy"


def test_retries_a_reading_within_its_patience() -> None:
    routes = {("POST", "/release"): answers((200, results(("a", "in_use"))), (200, results(("a", "stopped"))))}
    with FakeService(routes).running() as fake:
        actions = TtsSuite("tts", fake.url, busy_wait_s=5).release(held("a"), reason="r",
                                                                      deadline=time.monotonic() + 60)
    assert actions[0].outcome == "released"
