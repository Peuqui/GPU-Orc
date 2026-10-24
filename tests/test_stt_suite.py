import time
from typing import Any

import pytest
from fakes import GPUS, FakeService, Route, answers

from gpu_orc.consumers import stt_suite
from gpu_orc.consumers.stt_suite import SttSuite
from gpu_orc.model import ConsumerError, Holder

UNLOAD = ("POST", "/unload?device=cuda")
FORCE = ("POST", "/unload?device=cuda&force=1")


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stt_suite, "POLL_S", 0.05)
    monkeypatch.setattr(stt_suite, "FORCE_MARGIN_S", 0.1)


def status(**fields: Any) -> dict[tuple[str, str], Route]:
    return {("GET", "/status"): answers((200, fields))}


def worker() -> list[Holder]:
    with FakeService(status(gpu_loaded=True, gpu_uuid="GPU-bbbb1111")).running() as fake:
        return SttSuite("stt", fake.url).holders(GPUS)


def release(routes: dict[tuple[str, str], Route], **settings: Any) -> tuple[list[Any], list[tuple[str, str, Any]]]:
    with FakeService(routes).running() as fake:
        actions = SttSuite("stt", fake.url, **settings).release(worker(), reason="r", deadline=time.monotonic() + 60)
        return actions, fake.calls


def test_holders_name_cards_and_footprint() -> None:
    routes = status(gpu_loaded=True, gpu_uuid="GPU-bbbb1111", gpu_footprint_mib=3100, gpu_busy=True,
                    diarize_loaded=True, diarize_gpu_uuid="GPU-cccc2222")
    with FakeService(routes).running() as fake:
        holders = SttSuite("stt", fake.url).holders(GPUS)
    assert [(h.item, h.gpu_uuids, dict(h.memory_mib), h.busy) for h in holders] == [
        ("gpu-worker", frozenset({"GPU-bbbb1111"}), {"GPU-bbbb1111": 3100}, True),
        ("diarize-worker", frozenset({"GPU-cccc2222"}), {}, False),
    ]


def test_no_worker_no_holder() -> None:
    with FakeService(status(gpu_loaded=False, diarize_loaded=False)).running() as fake:
        assert SttSuite("stt", fake.url).holders(GPUS) == []


def test_running_worker_without_uuid_is_an_error_not_a_guess() -> None:
    service = FakeService(status(gpu_loaded=True))
    with service.running() as fake, pytest.raises(ConsumerError, match="too old"):
        SttSuite("stt", fake.url).holders(GPUS)


def test_release_when_idle() -> None:
    actions, _ = release({UNLOAD: answers((200, {"unloaded": ["gpu"]}))})
    assert [a.outcome for a in actions] == ["released"]


def test_waits_while_busy_then_released() -> None:
    routes = {UNLOAD: answers((409, {"busy": True}), (409, {"busy": True}), (200, {"unloaded": ["gpu"]}))}
    actions, calls = release(routes, busy_wait_s=5)
    assert sum(1 for call in calls if call[:2] == UNLOAD) == 3
    assert actions[0].outcome == "released"


def test_forced_after_its_patience_when_allowed() -> None:
    routes = {UNLOAD: answers((409, {"busy": True})), FORCE: answers((200, {"unloaded": ["gpu"]}))}
    actions, calls = release(routes, busy_wait_s=0.2, force_after_wait=True)
    assert any(call[:2] == FORCE for call in calls)
    assert actions[0].outcome == "released_forced"
    assert actions[0].waited_s is not None and actions[0].waited_s >= 0.2


def test_refused_after_its_patience_without_force() -> None:
    actions, calls = release({UNLOAD: answers((409, {"busy": True}))}, busy_wait_s=0.2)
    assert not any(call[:2] == FORCE for call in calls)
    assert actions[0].outcome == "refused_busy"


def test_overall_limit_cuts_the_patience_short() -> None:
    holders = worker()
    with FakeService({UNLOAD: answers((409, {"busy": True}))}).running() as fake:
        started = time.monotonic()
        actions = SttSuite("stt", fake.url, busy_wait_s=60).release(holders, reason="r", deadline=started + 0.5)
    assert actions[0].outcome == "timeout"
    assert time.monotonic() - started < 2


def test_unexpected_status_is_an_error() -> None:
    with pytest.raises(ConsumerError, match="500"):
        release({UNLOAD: answers((500, {"error": "boom"}))})
