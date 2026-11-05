import time
from typing import Any

import pytest
from fakes import GPUS, FakeService, Route, answers

from gpu_orc.consumers import tts_suite_remote
from gpu_orc.consumers.tts_suite_remote import TtsSuiteRemote
from gpu_orc.model import Action, ConsumerError, Holder

RELEASE = ("POST", "/release")
ENGINES = [
    {"engine": "qwen3local", "host": None, "status": "running"},  # on the suite's own machine
    {"engine": "qwen3local", "host": "Box", "status": "running"},
    {"engine": "xtts", "host": "Box", "status": "starting"},
    {"engine": "piper", "host": "Box", "status": "stopped"},
    {"engine": "xtts", "host": "Other", "status": "running"},
]


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tts_suite_remote, "POLL_S", 0.05)


def released(*pairs: tuple[str, str]) -> tuple[int, Any]:
    return 200, [{"engine": engine, "host": "Box", "outcome": outcome, "detail": ""} for engine, outcome in pairs]


def consumer(url: str, **settings: Any) -> TtsSuiteRemote:
    return TtsSuiteRemote("tts-remote", url, host="Box", token="s3cret", **settings)


def release(routes: dict[tuple[str, str], Route], *engines: str, **settings: Any) -> tuple[list[Action], FakeService]:
    with FakeService(routes).running() as fake:
        holders = [Holder("tts-remote", engine, frozenset(g.uuid for g in GPUS)) for engine in engines]
        return consumer(fake.url, **settings).release(holders, reason="hold", deadline=time.monotonic() + 30), fake


def test_running_and_starting_containers_of_this_machine_hold_every_card() -> None:
    with FakeService({("GET", "/engines"): answers((200, ENGINES))}).running() as fake:
        holders = consumer(fake.url).holders(GPUS)
        token = fake.headers[0].get("x-tts-suite-token")
    assert [(h.item, h.gpu_uuids) for h in holders] == [
        ("qwen3local", frozenset(g.uuid for g in GPUS)), ("xtts", frozenset(g.uuid for g in GPUS)),
    ]
    assert token == "s3cret"


def test_a_suite_that_refuses_is_an_error() -> None:
    with FakeService({("GET", "/engines"): answers((401, {"detail": "unauthorized"}))}).running() as fake, \
            pytest.raises(ConsumerError, match="HTTP 401"):
        consumer(fake.url).holders(GPUS)


def test_stopped_containers_are_released() -> None:
    actions, fake = release({RELEASE: answers(released(("qwen3local", "stopped"), ("xtts", "not_running")))},
                            "qwen3local", "xtts")
    assert [(a.item, a.outcome) for a in actions] == [("qwen3local", "released"), ("xtts", "released")]
    assert fake.calls[0][2] == {"targets": [{"engine": "qwen3local", "host": "Box"}, {"engine": "xtts", "host": "Box"}]}
    assert fake.headers[0].get("x-tts-suite-token") == "s3cret"


def test_a_container_a_reading_holds_is_waited_for() -> None:
    actions, fake = release({RELEASE: answers(released(("xtts", "in_use")), released(("xtts", "stopped")))},
                            "xtts", busy_wait_s=5)
    assert [(a.item, a.outcome) for a in actions] == [("xtts", "released")]
    assert len(fake.calls) == 2


def test_a_reading_past_the_wait_is_refused_busy() -> None:
    actions, _ = release({RELEASE: answers(released(("xtts", "in_use")))}, "xtts", busy_wait_s=0.2)
    assert [(a.item, a.outcome) for a in actions] == [("xtts", "refused_busy")]


def test_a_failed_stop_is_not_freed() -> None:
    actions, _ = release({RELEASE: answers(released(("xtts", "failed")))}, "xtts")
    assert [(a.item, a.outcome) for a in actions] == [("xtts", "not_freed")]


def test_an_answer_missing_a_target_is_an_error() -> None:
    with pytest.raises(ConsumerError, match="did not answer"):
        release({RELEASE: answers(released(("xtts", "stopped")))}, "xtts", "qwen3local")
