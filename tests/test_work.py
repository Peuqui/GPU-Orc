from typing import Any

import pytest

from gpu_orc.orchestrator import RequestError
from gpu_orc.server import parse_work_request
from gpu_orc.work import WorkStore


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def owners(store: WorkStore) -> list[str]:
    return [str(work["owner"]) for work in store.active()]


def test_reported_work_shows_until_it_ends() -> None:
    store = WorkStore(Clock())
    store.begin("job-1", "nightly digest", 300)
    assert owners(store) == ["job-1"]
    assert store.end("job-1") is True
    assert store.active() == []


def test_a_lease_that_is_not_renewed_ends_by_itself() -> None:
    clock = Clock()
    store = WorkStore(clock)
    store.begin("job-1", "digest", 300)
    clock.now += 299
    assert owners(store) == ["job-1"]
    clock.now += 2
    assert store.active() == []


def test_renewing_keeps_the_start_and_moves_the_end() -> None:
    clock = Clock()
    store = WorkStore(clock)
    store.begin("job-1", "digest", 300)
    clock.now += 200
    store.begin("job-1", "digest, step 2", 300)
    clock.now += 200  # 400 s after the first report: alive only because of the renewal
    (work,) = store.active()
    assert (work["reason"], work["running_for_s"], work["expires_in_s"]) == ("digest, step 2", 400.0, 100.0)


def test_ending_work_that_is_gone_says_so() -> None:
    clock = Clock()
    store = WorkStore(clock)
    store.begin("job-1", "digest", 10)
    clock.now += 11
    assert store.end("job-1") is False


def test_owners_are_independent() -> None:
    store = WorkStore(Clock())
    store.begin("job-1", "a", 300)
    store.begin("chat-7", "b", 300)
    store.end("job-1")
    assert owners(store) == ["chat-7"]


@pytest.mark.parametrize("body", [
    None, [], {}, {"owner": "o", "reason": "r"}, {"owner": "", "reason": "r", "ttl_s": 5},
    {"owner": "o", "reason": " ", "ttl_s": 5}, {"owner": "o", "reason": "r", "ttl_s": 0},
    {"owner": "o", "reason": "r", "ttl_s": -1}, {"owner": "o", "reason": "r", "ttl_s": True},
    {"owner": "o", "reason": "r", "ttl_s": "5"}, {"owner": "o", "reason": "r", "ttl_s": 5, "extra": 1},
])
def test_parse_work_rejects(body: Any) -> None:
    with pytest.raises(RequestError):
        parse_work_request(body)


def test_parse_work_accepts() -> None:
    assert parse_work_request({"owner": " job 1 ", "reason": "digest", "ttl_s": 300}) == ("job 1", "digest", 300.0)
