"""The Python client against the real GpuOrcServer (fake services behind it)."""

from __future__ import annotations

import socket
import sys
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from client_setup import machine
from fakes import serving

from gpu_orc.gpus import NVIDIA_SMI_TIMEOUT_S
from gpu_orc.http_json import REQUEST_TIMEOUT_S

sys.path.insert(0, str(Path(__file__).parent.parent / "clients" / "python"))
from gpu_orc_client import LOADING_TIMEOUT_S, STATUS_TIMEOUT_S, GpuOrc, GpuOrcError


@pytest.fixture
def orc() -> Iterator[GpuOrc]:
    with serving(machine()) as port:
        yield GpuOrc(f"http://127.0.0.1:{port}/")


@pytest.fixture
def stubborn_orc() -> Iterator[GpuOrc]:
    with serving(machine(stubborn=True)) as port:
        yield GpuOrc(f"http://127.0.0.1:{port}")


def test_model_that_already_runs(orc: GpuOrc) -> None:
    result = orc.free_for_model("big", "test")
    assert result.already_running and result.free and result.actions == () and result.problems() == []


def test_free_with_a_release(orc: GpuOrc) -> None:
    result = orc.free_for_model("small-vlm", "test")
    assert not result.already_running and result.free and result.problems() == []
    assert [(a.consumer, a.item, a.outcome) for a in result.actions] == [("stt", "gpu-worker", "released")]


def test_busy_consumer_is_a_problem(stubborn_orc: GpuOrc) -> None:
    result = stubborn_orc.free_for_model("elsewhere", "test")
    assert not result.free
    assert result.problems() == ["tts qwen: refused_busy (still busy)", "the cards are not free"]


def test_free_gpus_all_with_keep(stubborn_orc: GpuOrc) -> None:
    result = stubborn_orc.free_gpus("all", "test", keep=["ls"])
    assert {(a.consumer, a.item): a.outcome for a in result.actions} == {
        ("stt", "gpu-worker"): "released", ("ls", "big"): "kept", ("tts", "qwen"): "refused_busy",
    }
    assert result.problems() == ["tts qwen: refused_busy (still busy)", "the cards are not free"]


def test_free_gpus_by_index(orc: GpuOrc) -> None:
    result = orc.free_gpus([1], "test")
    assert result.free and [a.outcome for a in result.actions] == ["released"]


def test_unknown_model_is_an_error(orc: GpuOrc) -> None:
    with pytest.raises(GpuOrcError, match="HTTP 400"):
        orc.free_for_model("ghost", "test")


def test_traffic_light_green_then_red_while_a_model_loads(orc: GpuOrc) -> None:
    assert not orc.loading().red
    orc.free_for_model("small-vlm", "test")  # clears for the model: it counts as loading for the grace time
    light = orc.loading()
    assert light.red and light.loading == ("small-vlm",) and light.held is None


def test_hold_makes_the_light_red_until_released() -> None:
    with serving(machine()) as port:
        orc = GpuOrc(f"http://127.0.0.1:{port}")
        result = orc.hold("gaming")
        assert result.free and [a.outcome for a in result.actions] == ["released", "released"]
        light = orc.loading()
        assert light.red and light.loading == () and light.held is not None and light.held["reason"] == "gaming"
        orc.release_hold()
        assert not orc.loading().red


def test_work_is_visible_in_status(orc: GpuOrc) -> None:
    orc.work_begin("aifred/job 1", "digest", 60)
    assert [w["owner"] for w in orc.status()["work"]] == ["aifred/job 1"]
    orc.work_end("aifred/job 1")
    assert orc.status()["work"] == []
    orc.work_end("aifred/job 1")  # already gone: no error


def test_status_is_the_raw_answer(orc: GpuOrc) -> None:
    status = orc.status()
    assert len(status["gpus"]) == 3 and status["errors"] == []


def lan_address() -> str | None:
    """An address of this machine that is not loopback, where GPU-Orc asks for the token (None: no network)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("192.0.2.1", 9))  # no packet is sent
            address = str(probe.getsockname()[0])
        except OSError:
            return None
    return None if address.startswith("127.") else address


def test_token() -> None:
    address = lan_address()
    if address is None:
        pytest.skip("no non-loopback address: GPU-Orc lets loopback pass without a token")
    with serving(machine(), token="secret", bind="0.0.0.0") as port:
        url = f"http://{address}:{port}"
        with pytest.raises(GpuOrcError, match="HTTP 401"):
            GpuOrc(url).loading()
        with pytest.raises(GpuOrcError, match="HTTP 403"):
            GpuOrc(url, "wrong").loading()
        assert not GpuOrc(url, "secret").loading().red
        assert GpuOrc(url, "secret").free_for_model("big", "test").already_running


def test_unreachable() -> None:
    with pytest.raises(GpuOrcError, match="not reachable"):
        GpuOrc("http://127.0.0.1:9").loading()
    with pytest.raises(GpuOrcError, match=r"not reachable at http://127\.0\.0\.1:9 \(/loading\)"):
        GpuOrc("http://127.0.0.1:9/").loading()
    with pytest.raises(GpuOrcError, match="not reachable"):
        GpuOrc("http://127.0.0.1:9").free_for_model("big", "test")


def test_model_server_is_sent_only_when_given() -> None:
    with serving(machine(two_servers=True)) as port:
        orc = GpuOrc(f"http://127.0.0.1:{port}")
        with pytest.raises(GpuOrcError, match="HTTP 400"):  # several model servers: the name is required
            orc.free_for_model("big", "test")
        with pytest.raises(GpuOrcError, match="HTTP 404"):
            orc.free_for_model("big", "test", model_server="nope")
        assert orc.free_for_model("big", "test", model_server="ls").already_running
        assert not orc.free_for_model("big", "test", model_server="vllm").already_running


def test_free_waits_as_long_as_gpu_orc_says_and_status_as_long_as_it_needs(
    orc: GpuOrc, monkeypatch: pytest.MonkeyPatch
) -> None:
    caller_timeout_s = orc._call("GET", "/health", 5)["caller_timeout_s"]
    waits: dict[str, float] = {}
    real_urlopen = urllib.request.urlopen

    def recording_urlopen(request: urllib.request.Request, timeout: float) -> Any:
        waits[request.full_url.rsplit("/", 1)[1]] = timeout
        return real_urlopen(request, timeout=timeout)

    monkeypatch.setattr(urllib.request, "urlopen", recording_urlopen)
    orc.free_for_model("small-vlm", "test")
    orc.status()
    orc.loading()
    assert waits["free"] == caller_timeout_s
    assert waits["status"] == STATUS_TIMEOUT_S
    # A slow traffic light counts as red: the caller then puts nothing on a card instead of waiting.
    assert waits["loading"] == LOADING_TIMEOUT_S < STATUS_TIMEOUT_S


def test_a_traffic_light_of_an_unknown_form_is_an_error(
    orc: GpuOrc, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(GpuOrc, "_call", lambda *args, **kwargs: {"loading": []})  # no "held"
    with pytest.raises(GpuOrcError, match="unknown form"):
        orc.loading()


def test_status_timeout_covers_the_servers_own_limits() -> None:
    """/status asks nvidia-smi, then every consumer in parallel."""
    assert STATUS_TIMEOUT_S > NVIDIA_SMI_TIMEOUT_S + REQUEST_TIMEOUT_S
