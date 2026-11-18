"""The Python client against the real GpuOrcServer (fake services behind it)."""

from __future__ import annotations

import socket
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from client_setup import machine
from fakes import serving

sys.path.insert(0, str(Path(__file__).parent.parent / "clients" / "python"))
from gpu_orc_client import GpuOrc, GpuOrcError


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
