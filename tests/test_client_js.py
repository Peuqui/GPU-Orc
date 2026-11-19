"""The JS client against the real GpuOrcServer, run through node (clients/js/check_against_server.mjs)."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from client_setup import machine
from fakes import serving

CHECK = Path(__file__).parent.parent / "clients" / "js" / "check_against_server.mjs"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def node(*args: str) -> Any:
    done = subprocess.run(["node", str(CHECK), *args], capture_output=True, text=True, timeout=60, check=True)
    return json.loads(done.stdout)


def test_free_for_a_model_that_already_runs() -> None:
    with serving(machine()) as port:
        assert node(f"http://127.0.0.1:{port}", "free", "big") == {
            "free": True, "alreadyRunning": True, "actions": [], "problems": [],
        }


def test_free_with_a_release_then_the_light_shows_the_model_loading() -> None:
    with serving(machine()) as port:
        url = f"http://127.0.0.1:{port}"
        assert node(url, "loading") == {"loading": [], "held": None, "holdRequested": None, "red": False}
        result = node(url, "free", "small-vlm")
        assert result["free"] is True and result["problems"] == []
        assert [(a["consumer"], a["item"], a["outcome"]) for a in result["actions"]] == [("stt", "gpu-worker", "released")]
        assert node(url, "loading") == {"loading": ["small-vlm"], "held": None, "holdRequested": None, "red": True}


def test_busy_consumer_gives_the_same_problems_as_python() -> None:
    with serving(machine(stubborn=True)) as port:
        result = node(f"http://127.0.0.1:{port}", "free", "elsewhere")
        assert result["free"] is False
        assert result["problems"] == ["tts qwen: refused_busy (still busy)", "the cards are not free"]


def test_keep_is_passed_on() -> None:
    with serving(machine()) as port:
        result = node(f"http://127.0.0.1:{port}", "free", "small-vlm", "stt")
        assert [(a["consumer"], a["outcome"]) for a in result["actions"]] == [("stt", "kept")]


def test_hold_shows_red() -> None:
    with serving(machine()) as port:
        url = f"http://127.0.0.1:{port}"
        request = urllib.request.Request(f"{url}/hold", data=json.dumps({"owner": "game", "reason": "gaming"}).encode(), method="POST")
        urllib.request.urlopen(request, timeout=10).read()
        deadline = time.monotonic() + 5
        while (light := node(url, "loading"))["held"] is None:  # the hold worker clears, then the hold is active
            assert time.monotonic() < deadline
            time.sleep(0.05)
        assert light["red"] is True and light["held"]["reason"] == "gaming" and light["held"]["owner"] == "game"
        assert light["holdRequested"] is None


def test_hold_requested_shows_but_the_light_stays_green() -> None:
    with serving(machine()) as port:
        url = f"http://127.0.0.1:{port}"
        for path, body in (("/work", {"owner": "aifred", "reason": "turn", "ttl_s": 60}),
                           ("/hold", {"owner": "game", "reason": "gaming"})):
            request = urllib.request.Request(f"{url}{path}", data=json.dumps(body).encode(), method="POST")
            urllib.request.urlopen(request, timeout=10).read()
        light = node(url, "loading")
        assert light["red"] is False and light["held"] is None
        assert light["holdRequested"]["owner"] == "game" and light["holdRequested"]["state"] == "requested"


def test_errors_are_gpu_orc_errors() -> None:
    with serving(machine()) as port:
        refused = node(f"http://127.0.0.1:{port}", "free", "ghost")
        assert refused["error"] == "GpuOrcError" and "HTTP 400" in refused["message"]
    unreachable = node("http://127.0.0.1:9", "loading")
    assert unreachable["error"] == "GpuOrcError" and "not reachable" in unreachable["message"]
