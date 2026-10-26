import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http import HTTPStatus
from typing import Any

import pytest
from fakes import GPUS
from test_orchestrator import FakeService, model, server

from gpu_orc.orchestrator import Orchestrator, RequestError
from gpu_orc.server import UI_DIR, GpuOrcServer, authorization_error, parse_free_request, ui_file


@pytest.fixture
def url() -> Iterator[str]:
    consumers = [FakeService("stt"), server(model("big", GPUS[0].uuid))]
    httpd = GpuOrcServer(("127.0.0.1", 0), Orchestrator(consumers, 660, query=lambda: list(GPUS)), token=None)
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


def test_profiles_and_needs(url: str) -> None:
    status, body = call(f"{url}/profiles")
    assert status == 200 and [profile["name"] for profile in body["profiles"]] == ["big", "small-vlm", "elsewhere"]
    assert call(f"{url}/needs") == (200, {"needs": [], "errors": []})


def fetch(url: str) -> tuple[int, str, str, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status, response.headers["Content-Type"], response.geturl(), response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers["Content-Type"], url, error.read()


def test_the_page_is_served_and_the_root_leads_to_it(url: str) -> None:
    status, kind, final, body = fetch(f"{url}/")
    assert (status, kind, final) == (200, "text/html; charset=utf-8", f"{url}/ui/")
    assert b"<title>GPU-Orc</title>" in body
    status, kind, _, body = fetch(f"{url}/ui/app.js")
    assert (status, kind) == (200, "text/javascript; charset=utf-8") and b"import" in body
    assert fetch(f"{url}/ui/..%2Fserver.py")[0] == 404


@pytest.mark.parametrize("name", ["../server.py", "sub/app.js", ".hidden.js", "README.md", "missing.js", ""])
def test_only_files_of_the_page_are_served(name: str) -> None:
    assert ui_file(name) is None


def test_every_file_the_page_loads_is_there() -> None:
    for name in ("index.html", "style.css", "app.js", "api.js", "dom.js", "format.js", "i18n.js", "cards.js",
                 "profiles.js", "needs.js"):
        assert ui_file(name) == UI_DIR / name
