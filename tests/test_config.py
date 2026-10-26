from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_orc.config import ConfigError, load_config
from gpu_orc.consumers.gpu_release import GpuRelease
from gpu_orc.consumers.llama_swap import LlamaSwap

CONSUMERS: list[dict[str, Any]] = [
    {"name": "stt", "type": "gpu-release", "url": "http://127.0.0.1:5080/", "busy_wait_s": 600, "force_after_wait": True},
    {"name": "tts", "type": "gpu-release", "url": "http://127.0.0.1:5090", "busy_wait_s": 20},
    {"name": "ls", "type": "llama-swap", "url": "http://127.0.0.1:11435", "config_file": "~/ls.yaml",
     "no_eviction_models": ["*-vlm-*"]},
]


def write(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_full_config(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, {"consumers": CONSUMERS, "verify_timeout_s": 20, "min_freed_mib": 300}))
    assert (config.host, config.port, config.token, config.free_timeout_s) == ("127.0.0.1", 5070, None, 180.0)
    assert (config.verify.timeout_s, config.verify.min_freed_mib) == (20.0, 300)
    stt, _, ls = config.consumers
    assert isinstance(stt, GpuRelease) and isinstance(ls, LlamaSwap)
    assert (stt.url, stt.busy_wait_s, stt.force_after_wait) == ("http://127.0.0.1:5080", 600.0, True)
    assert str(ls.config_file).endswith("/ls.yaml") and "~" not in str(ls.config_file)


def test_network_listen_needs_a_token(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="token_file"):
        load_config(write(tmp_path, {"listen": {"host": "0.0.0.0"}, "consumers": CONSUMERS}))
    token = tmp_path / "token"
    token.write_text("secret\n")
    data = {"listen": {"host": "0.0.0.0"}, "token_file": str(token), "consumers": CONSUMERS}
    assert load_config(write(tmp_path, data)).token == "secret"


@pytest.mark.parametrize("consumers, message", [
    ([{"name": "x", "type": "ollama", "url": "http://h"}], "unknown type"),
    ([{"name": "x", "type": "gpu-release", "url": "http://h", "force": True}], "unknown keys"),
    ([{"name": "x", "type": "gpu-release"}], "url is required"),
    ([CONSUMERS[1], CONSUMERS[1]], "unique"),
    ([], "non-empty"),
])
def test_rejected_consumers(tmp_path: Path, consumers: list[dict[str, Any]], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(write(tmp_path, {"consumers": consumers}))


def test_a_port_is_no_boolean(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="wrong type"):
        load_config(write(tmp_path, {"listen": {"port": True}, "consumers": CONSUMERS}))


def test_state_dir_and_vision_models(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, {"consumers": CONSUMERS}))
    assert str(config.state_dir).endswith("/.local/state/gpu-orc") and "~" not in str(config.state_dir)
    assert config.vision_models == {}
    data = {"consumers": CONSUMERS, "state_dir": str(tmp_path / "state"), "vision_models": {"ls": ["Qwen3VL-4B"]}}
    config = load_config(write(tmp_path, data))
    assert config.state_dir == tmp_path / "state" and config.vision_models == {"ls": ("Qwen3VL-4B",)}


@pytest.mark.parametrize("vision_models, message", [
    ({"tts": ["x"]}, "not a configured model server"),
    ({"ls": "Qwen3VL-4B"}, "list of model names"),
    (["Qwen3VL-4B"], "mapping"),
])
def test_rejected_vision_models(tmp_path: Path, vision_models: Any, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(write(tmp_path, {"consumers": CONSUMERS, "vision_models": vision_models}))
