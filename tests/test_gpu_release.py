import time
from typing import Any

import pytest
from fakes import GPUS, FakeService, Route, answers

from gpu_orc import http_json
from gpu_orc.consumers import gpu_release
from gpu_orc.consumers.gpu_release import GpuRelease
from gpu_orc.model import Action, ConsumerError, Holder, Need

RELEASE = ("POST", "/gpu/release")


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_release, "POLL_S", 0.05)
    monkeypatch.setattr(gpu_release, "FORCE_MARGIN_S", 0.1)


def results(*pairs: tuple[str, str], status: int = 200) -> tuple[int, Any]:
    return status, {"results": [{"item": item, "outcome": outcome, "detail": ""} for item, outcome in pairs]}


def held(*items: str) -> list[Holder]:
    return [Holder("svc", item, frozenset({"GPU-aaaa0000"})) for item in items]


def release(routes: dict[tuple[str, str], Route], *items: str, deadline_s: float = 60,
            **settings: Any) -> tuple[list[Action], list[tuple[str, str, Any]]]:
    with FakeService(routes).running() as fake:
        actions = GpuRelease("svc", fake.url, **settings).release(
            held(*items), reason="loading X", deadline=time.monotonic() + deadline_s)
        return actions, fake.calls


def test_holders_come_from_the_service_with_known_cards() -> None:
    body = {"holders": [
        {"item": "gpu-worker", "gpu_uuids": ["GPU-bbbb1111"], "memory_mib": {"GPU-bbbb1111": 3100}, "busy": True},
        {"item": "qwen", "gpu_uuids": ["GPU-cccc2222"], "memory_mib": None, "busy": False},
    ]}
    with FakeService({("GET", "/gpu/holders"): answers((200, body))}).running() as fake:
        holders = GpuRelease("svc", fake.url).holders(GPUS)
    assert [(h.item, h.gpu_uuids, dict(h.memory_mib), h.busy) for h in holders] == [
        ("gpu-worker", frozenset({"GPU-bbbb1111"}), {"GPU-bbbb1111": 3100}, True),
        ("qwen", frozenset({"GPU-cccc2222"}), {}, False),
    ]


def test_a_card_that_is_not_on_this_machine_is_an_error() -> None:
    body = {"holders": [{"item": "x", "gpu_uuids": ["GPU-elsewhere"], "memory_mib": None, "busy": False}]}
    service = FakeService({("GET", "/gpu/holders"): answers((200, body))})
    with service.running() as fake, pytest.raises(ConsumerError, match="not on this machine"):
        GpuRelease("svc", fake.url).holders(GPUS)


def test_released_and_not_running() -> None:
    actions, calls = release({RELEASE: answers(results(("a", "released"), ("b", "not_running")))}, "a", "b")
    assert [(a.item, a.outcome) for a in actions] == [("a", "released"), ("b", "released")]
    assert calls[0][2] == {"items": ["a", "b"], "force": False, "reason": "loading X"}


def test_waits_while_busy_then_released() -> None:
    routes = {RELEASE: answers(results(("a", "busy"), status=409), results(("a", "released")))}
    actions, calls = release(routes, "a", busy_wait_s=5)
    assert [a.outcome for a in actions] == ["released"] and len(calls) == 2


def test_forced_after_its_patience_when_allowed() -> None:
    def route(body: Any) -> tuple[int, Any]:
        return results(("a", "released")) if body["force"] else results(("a", "busy"), status=409)

    actions, calls = release({RELEASE: route}, "a", busy_wait_s=0.2, force_after_wait=True)
    assert [a.outcome for a in actions] == ["released_forced"]
    assert calls[-1][2]["force"] is True


def test_refused_after_its_patience_without_force() -> None:
    actions, calls = release({RELEASE: answers(results(("a", "busy"), status=409))}, "a", busy_wait_s=0.2)
    assert [a.outcome for a in actions] == ["refused_busy"]
    assert not any(call[2]["force"] for call in calls)


def test_overall_limit_cuts_the_patience_short() -> None:
    started = time.monotonic()
    actions, _ = release({RELEASE: answers(results(("a", "busy"), status=409))}, "a", deadline_s=0.5, busy_wait_s=60)
    assert [a.outcome for a in actions] == ["timeout"] and time.monotonic() - started < 2


def test_a_release_may_answer_until_the_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    # A forced release stops a container and proves it down, maybe with docker kill: more than a plain 10 s question.
    timeouts: list[float | None] = []
    real = http_json.request_json

    def request_json(method: str, url: str, body: Any = None, **kwargs: Any) -> tuple[int, Any]:
        timeouts.append(kwargs.get("timeout"))
        return real(method, url, body, **kwargs)

    monkeypatch.setattr(gpu_release, "request_json", request_json)
    release({RELEASE: answers(results(("a", "released")))}, "a", deadline_s=300)
    assert len(timeouts) == 1 and timeouts[0] is not None and 290 < timeouts[0] <= 300


def test_failed_end_is_not_freed() -> None:
    reply = 500, {"results": [{"item": "a", "outcome": "failed", "detail": "container still running"}]}
    actions, _ = release({RELEASE: answers(reply)}, "a")
    assert actions[0].outcome == "not_freed" and "container still running" in actions[0].detail


def test_missing_answer_is_an_error() -> None:
    with pytest.raises(ConsumerError, match="did not answer"):
        release({RELEASE: answers(results(("other", "released")))}, "a")


def test_needs_come_from_the_service() -> None:
    body = {"needs": [
        {"item": "parakeet-fp32", "need_mib": 3712, "source": "measured", "default": True},
        {"item": "xtts", "need_mib": None, "source": "unknown"},
    ]}
    with FakeService({("GET", "/gpu/needs"): answers((200, body))}).running() as fake:
        assert GpuRelease("svc", fake.url).needs() == [
            Need("svc", "parakeet-fp32", 3712, "measured", True), Need("svc", "xtts", None, "unknown", False),
        ]


@pytest.mark.parametrize("answer", [
    (404, {"error": "no route"}),
    (200, {"needs": [{"item": "x"}]}),
    (200, {"needs": [{"item": "x", "need_mib": 1, "source": "guessed"}]}),
])
def test_needs_in_another_form_are_an_error(answer: tuple[int, Any]) -> None:
    with FakeService({("GET", "/gpu/needs"): answers(answer)}).running() as fake, pytest.raises(ConsumerError):
        GpuRelease("svc", fake.url).needs()
