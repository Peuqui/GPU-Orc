import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
from http import HTTPStatus
from pathlib import Path
from typing import Any

import pytest
from fakes import GPUS
from test_orchestrator import FAST, FakeService, Machine, model, server

from gpu_orc.hold import HoldStore
from gpu_orc.idle_run import run_if_idle
from gpu_orc.orchestrator import Orchestrator, RequestError
from gpu_orc.server import GpuOrcServer, Handler, authorization_error, parse_free_request
from gpu_orc.work import WorkStore


@pytest.fixture
def url(tmp_path: Path) -> Iterator[str]:
    memory = Machine()
    consumers = [FakeService("stt"), server(model("big", GPUS[0].uuid), machine=memory)]
    orchestrator = Orchestrator(consumers, 660, query=memory.query, verify_settings=FAST,
                                hold=HoldStore(tmp_path / "hold.json"), work=WorkStore())
    orchestrator.start_hold_worker()
    httpd = GpuOrcServer(("127.0.0.1", 0), orchestrator, token=None)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def call(url: str, method: str = "GET", body: Any = None) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method=method)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_health_names_the_timeouts(url: str) -> None:
    status, body = call(f"{url}/health")
    assert status == 200 and body["free_timeout_s"] == 660 and body["caller_timeout_s"] == 690


def test_status(url: str) -> None:
    status, body = call(f"{url}/status")
    assert status == 200 and len(body["gpus"]) == 3 and body["errors"] == []


def test_status_carries_the_handover_and_loading_does_not(quiet_url: str) -> None:
    url = quiet_url
    assert call(f"{url}/status")[1]["handover"] is None
    call(f"{url}/hold", "POST", {"owner": "game", "reason": "gaming", "model": None})
    held_over_http(url)
    call(f"{url}/hold/end", "POST", {"owner": "game", "by": "peuqui"})
    assert call(f"{url}/hold", "DELETE", {"owner": "game"}) == (200, {"hold": None})
    handover = call(f"{url}/status")[1]["handover"]
    assert set(handover) == {"for", "since", "seconds_left"} and handover["for"] == "peuqui"
    assert 0 < handover["seconds_left"] <= 120
    assert call(f"{url}/loading") == (200, {"loading": [], "held": None, "hold_requested": None})


def test_loading(url: str) -> None:
    assert call(f"{url}/loading") == (200, {"loading": [], "held": None, "hold_requested": None})


def test_free_model_already_running(url: str) -> None:
    status, body = call(f"{url}/free", "POST", {"model": "big", "reason": "hook"})
    assert status == 200 and body["already_running"] is True


@pytest.mark.parametrize("body, code", [
    ({"model": "big"}, 400),                                   # no reason
    ({"gpus": [0], "model": "big", "reason": "r"}, 400),        # both forms
    ({"model": "ghost", "reason": "r"}, 400),
    ({"model": "big", "model_server": "vllm", "reason": "r"}, 404),
])
def test_free_errors(url: str, body: dict[str, Any], code: int) -> None:
    assert call(f"{url}/free", "POST", body)[0] == code


def test_unknown_path(url: str) -> None:
    assert call(f"{url}/nope")[0] == 404


@pytest.mark.parametrize("body", [
    [], {"gpus": [], "reason": "r"}, {"gpus": [True], "reason": "r"}, {"gpus": "some", "reason": "r"},
    {"gpus": [0], "reason": "r", "model_server": "ls"}, {"gpus": [0], "reason": "r", "keep": "ls"},
    {"gpus": [0], "reason": "r", "dry_run": "yes"}, {"gpus": [0], "reason": "r", "extra": 1},
])
def test_parse_rejects(body: Any) -> None:
    with pytest.raises(RequestError):
        parse_free_request(body)


def test_parse_accepts_both_forms() -> None:
    assert parse_free_request({"gpus": "all", "reason": "r"}).gpus == "all"
    request = parse_free_request({"model": "m", "model_server": "ls", "keep": ["tts"], "reason": "r", "dry_run": True})
    assert (request.model, request.model_server, request.keep, request.dry_run) == ("m", "ls", ("tts",), True)


def test_authorization() -> None:
    assert authorization_error("127.0.0.1", "", None) is None
    assert authorization_error("::1", "", "t") is None
    assert authorization_error("203.0.113.7", "", "t") == (HTTPStatus.UNAUTHORIZED, "token required")
    assert authorization_error("203.0.113.7", "Bearer wrong", "t") == (HTTPStatus.FORBIDDEN, "wrong token")
    assert authorization_error("203.0.113.7", "Bearer t", "t") is None
    assert authorization_error("203.0.113.7", "Bearer t", None) == (HTTPStatus.FORBIDDEN, "wrong token")


@pytest.fixture
def quiet_url(tmp_path: Path) -> Iterator[str]:
    """A machine where nothing is held: a hold has nothing to clear and turns active at once."""
    orchestrator = Orchestrator([FakeService("stt"), server()], 660, query=lambda: list(GPUS),
                                hold=HoldStore(tmp_path / "hold.json"),
                                work=WorkStore())
    orchestrator.start_hold_worker()
    httpd = GpuOrcServer(("127.0.0.1", 0), orchestrator, token=None)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def held_over_http(url: str) -> Any:
    """The active hold, once the hold worker made it active."""
    deadline = time.monotonic() + 5
    while (held := call(f"{url}/loading")[1]["held"]) is None:
        assert time.monotonic() < deadline
        time.sleep(0.05)
    return held


def test_hold_and_let_go(quiet_url: str) -> None:
    url = quiet_url
    status, body = call(f"{url}/hold", "POST", {"owner": "game", "reason": "gaming", "model": None})
    requested = body["hold"]
    assert status == 200 and (requested["owner"], requested["reason"], requested["state"]) == ("game", "gaming", "requested")
    assert requested["free"] is None and requested["model"] is None and requested["model_server"] is None
    assert requested["granted_by"] is None and requested["end_requested_by"] is None
    hold = held_over_http(url)
    assert (hold["owner"], hold["since"], hold["state"]) == ("game", requested["since"], "active")
    assert "free" in hold and hold["free"]["free"] is True
    assert call(f"{url}/hold", "POST", {"owner": "game", "reason": "gaming", "model": None}) == (200, {"hold": hold})
    light = call(f"{url}/loading")[1]
    assert light["held"] == hold and light["hold_requested"] is None
    assert call(f"{url}/status")[1]["held"] == hold
    assert call(f"{url}/hold", "DELETE", {"owner": "game"}) == (200, {"hold": None})
    assert call(f"{url}/loading") == (200, {"loading": [], "held": None, "hold_requested": None})


def test_a_hold_with_a_loaded_main_model_is_requested_until_granted_over_http(url: str) -> None:
    status, body = call(f"{url}/hold", "POST", {"owner": "game", "reason": "gaming", "model": None})
    assert status == 200 and body["hold"]["state"] == "requested" and body["hold"]["free"] is None
    time.sleep(0.3)
    light = call(f"{url}/loading")[1]
    assert light["held"] is None and light["hold_requested"] == body["hold"]
    status, granted = call(f"{url}/hold/grant", "POST", {"owner": "game", "by": "peuqui"})
    assert status == 200 and granted["hold"]["granted_by"] == "peuqui" and granted["hold"]["state"] == "requested"
    assert held_over_http(url)["granted_by"] == "peuqui"
    assert call(f"{url}/hold/grant", "POST", {"owner": "game", "by": "peuqui"})[1] == {"hold": held_over_http(url)}


def test_a_grant_answers_404_without_a_hold_and_409_for_another_owner(url: str) -> None:
    assert call(f"{url}/hold/grant", "POST", {"owner": "game", "by": "peuqui"})[0] == 404
    call(f"{url}/hold", "POST", {"owner": "game", "reason": "gaming", "model": None})
    status, body = call(f"{url}/hold/grant", "POST", {"owner": "render", "by": "peuqui"})
    assert status == 409 and body["owner"] == "game" and body["state"] == "requested"


@pytest.mark.parametrize("body", [
    None, {}, {"owner": "o"}, {"by": "p"}, {"owner": "", "by": "p"}, {"owner": "o", "by": ""}, {"owner": "o", "by": 3},
    {"owner": "o", "by": "p", "x": 1},
])
def test_a_grant_needs_an_owner_and_a_by(url: str, body: Any) -> None:
    assert call(f"{url}/hold/grant", "POST", body)[0] == 400


def test_an_end_request_marks_an_active_hold_over_http(quiet_url: str) -> None:
    url = quiet_url
    call(f"{url}/hold", "POST", {"owner": "game", "reason": "gaming", "model": None})
    hold = held_over_http(url)
    status, body = call(f"{url}/hold/end", "POST", {"owner": "game", "by": "peuqui"})
    assert status == 200 and body == {"hold": {**hold, "end_requested_by": "peuqui"}}
    assert held_over_http(url)["end_requested_by"] == "peuqui"
    assert call(f"{url}/hold/end", "POST", {"owner": "game", "by": "other"})[1] == body
    assert call(f"{url}/hold", "DELETE", {"owner": "game"}) == (200, {"hold": None})


def test_an_end_request_withdraws_a_requested_hold_over_http(url: str) -> None:
    call(f"{url}/hold", "POST", {"owner": "game", "reason": "gaming", "model": None})
    assert call(f"{url}/loading")[1]["hold_requested"] is not None
    assert call(f"{url}/hold/end", "POST", {"owner": "game", "by": "peuqui"}) == (200, {"hold": None})
    assert call(f"{url}/loading") == (200, {"loading": [], "held": None, "hold_requested": None})


def test_an_end_request_answers_404_without_a_hold_and_409_for_another_owner(url: str) -> None:
    assert call(f"{url}/hold/end", "POST", {"owner": "game", "by": "peuqui"})[0] == 404
    call(f"{url}/hold", "POST", {"owner": "game", "reason": "gaming", "model": None})
    status, body = call(f"{url}/hold/end", "POST", {"owner": "render", "by": "peuqui"})
    assert status == 409 and body["owner"] == "game" and body["state"] == "requested"


@pytest.mark.parametrize("body", [
    None, {}, {"owner": "o"}, {"by": "p"}, {"owner": "", "by": "p"}, {"owner": "o", "by": ""}, {"owner": "o", "by": 3},
    {"owner": "o", "by": "p", "x": 1},
])
def test_an_end_request_needs_an_owner_and_a_by(url: str, body: Any) -> None:
    assert call(f"{url}/hold/end", "POST", body)[0] == 400


def test_a_hold_for_a_model_is_checked_and_keeps_it(url: str) -> None:
    status, body = call(f"{url}/hold", "POST", {"owner": "game", "reason": "gaming", "model": "ghost"})
    assert status == 400 and "ghost" in body["error"]
    assert call(f"{url}/loading")[1]["hold_requested"] is None
    status, body = call(f"{url}/hold", "POST", {"owner": "game", "reason": "gaming", "model": "big", "model_server": "ls"})
    assert status == 200 and (body["hold"]["model"], body["hold"]["model_server"]) == ("big", "ls")
    assert held_over_http(url)["free"]["actions"] == [{"consumer": "ls", "item": "big", "outcome": "kept"}]


def test_another_owner_gets_409_with_the_holder_and_cannot_release(quiet_url: str) -> None:
    url = quiet_url
    call(f"{url}/hold", "POST", {"owner": "game", "reason": "gaming", "model": None})
    held = held_over_http(url)
    status, body = call(f"{url}/hold", "POST", {"owner": "render", "reason": "job", "model": None})
    assert status == 409 and (body["owner"], body["state"], body["since"]) == ("game", "active", held["since"])
    assert "game" in body["error"]
    assert call(f"{url}/hold", "DELETE", {"owner": "render"})[0] == 409
    assert call(f"{url}/loading")[1]["held"] is not None
    assert call(f"{url}/hold", "POST", {"owner": "game", "reason": "again", "model": None}) == (200, {"hold": held})  # same owner: no change
    assert call(f"{url}/hold", "DELETE", {"owner": "person", "force": True}) == (200, {"hold": None})


@pytest.mark.parametrize("body", [
    None, {}, {"reason": "x", "model": None}, {"owner": "o", "model": None}, {"owner": "o", "reason": "x"},
    {"owner": "o", "reason": "", "model": None}, {"owner": "", "reason": "x", "model": None},
    {"owner": "o", "reason": 3, "model": None}, {"owner": "o", "reason": "x", "model": None, "until": 5},
    {"owner": "o", "reason": "x", "model": 5}, {"owner": "o", "reason": "x", "model": ""},
    {"owner": "o", "reason": "x", "model": None, "model_server": "ls"},
    {"owner": "o", "reason": "x", "model": "big", "model_server": ""},
])
def test_a_hold_needs_an_owner_a_reason_and_a_model_or_null(url: str, body: Any) -> None:
    status, answer = call(f"{url}/hold", "POST", body)
    assert status == 400 and ("owner" in answer["error"] or "model" in answer["error"])


@pytest.mark.parametrize("body", [None, {}, {"owner": ""}, {"owner": "o", "force": "yes"}, {"owner": "o", "x": 1}])
def test_a_release_needs_an_owner(url: str, body: Any) -> None:
    assert call(f"{url}/hold", "DELETE", body)[0] == 400


def test_work_over_http(url: str) -> None:
    assert call(f"{url}/work") == (200, {"work": []})
    status, body = call(f"{url}/work", "POST", {"owner": "aifred/job 12", "reason": "digest", "ttl_s": 300})
    assert status == 200 and [work["owner"] for work in body["work"]] == ["aifred/job 12"]
    assert [work["reason"] for work in call(f"{url}/status")[1]["work"]] == ["digest"]
    assert call(f"{url}/work/aifred%2Fjob%2012", "DELETE") == (200, {"ended": True, "work": []})
    assert call(f"{url}/work/aifred%2Fjob%2012", "DELETE") == (200, {"ended": False, "work": []})
    assert call(f"{url}/work", "POST", {"owner": "o"})[0] == 400
    assert call(f"{url}/work/", "DELETE")[0] == 404


def test_run_if_idle_runs_the_command_only_without_work(url: str) -> None:
    ran: list[list[str]] = []

    def run(command: Sequence[str]) -> int:
        ran.append(list(command))
        return 7

    assert run_if_idle(url, ["/sbin/reboot"], run) == 7 and ran == [["/sbin/reboot"]]
    call(f"{url}/work", "POST", {"owner": "aifred", "reason": "2 running turn(s)", "ttl_s": 60})
    assert run_if_idle(url, ["/sbin/reboot"], run) == 0 and len(ran) == 1
    call(f"{url}/work/aifred", "DELETE")
    assert run_if_idle(url, ["/sbin/reboot"], run) == 7 and len(ran) == 2


def test_run_if_idle_skips_the_command_when_gpu_orc_does_not_answer() -> None:
    ran: list[list[str]] = []
    assert run_if_idle("http://127.0.0.1:9", ["/sbin/reboot"], lambda command: ran.append(list(command))) == 3
    assert ran == []


def test_a_caller_hanging_up_is_no_nvidia_smi_failure() -> None:
    # 26.11.: llama-stats hung up while /status waited on a hanging consumer; the BrokenPipeError of the
    # send was taken for an nvidia-smi failure and a second answer was sent into the closed socket.
    handler = Handler.__new__(Handler)
    sent: list[HTTPStatus] = []

    def hung_up(status: HTTPStatus, body: Any) -> None:
        sent.append(status)
        raise BrokenPipeError(32, "Broken pipe")

    handler._send = hung_up  # type: ignore[method-assign]
    with pytest.raises(BrokenPipeError):
        handler._answer(lambda: {"gpus": []})
    assert sent == [HTTPStatus.OK]
