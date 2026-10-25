import time
from pathlib import Path
from typing import Any

import pytest
import yaml
from fakes import GPUS, FakeService, answers

from gpu_orc.consumers.llama_swap import LlamaSwap
from gpu_orc.gpus import GpuSpecError
from gpu_orc.model import ConsumerError, Holder, UnknownModel

CONFIG = {
    "models": {
        "main-a": {"cmd": "x", "env": ["CUDA_VISIBLE_DEVICES=GPU-aaaa0000,GPU-bbbb1111"]},
        "main-all": {"cmd": "x"},
        "main-vlm-b": {"cmd": "x", "env": ["CUDA_DEVICE_ORDER=PCI_BUS_ID", "CUDA_VISIBLE_DEVICES=1"]},
        "describer": {"cmd": "x", "env": ["CUDA_VISIBLE_DEVICES=GPU-bbbb1111"]},
        "embedder": {"cmd": "x", "env": ["CUDA_VISIBLE_DEVICES=GPU-cccc2222"]},
        "broken": {"cmd": "x", "env": ["CUDA_VISIBLE_DEVICES=3"]},
        "main-vllm": {"cmd": "python -m vllm.entrypoints.openai.api_server --gpu-memory-utilization 0.9 --port ${PORT}",
                      "env": ["CUDA_VISIBLE_DEVICES=GPU-cccc2222"]},
    },
    "groups": {
        "main": {"swap": True, "exclusive": True, "members": ["main-a", "main-all", "main-vlm-b"]},
        "vision": {"persistent": True, "members": ["describer"]},
        "embed": {"persistent": True, "members": ["embedder"]},
    },
}


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(CONFIG))
    return path


def running(*models: str) -> tuple[int, Any]:
    return 200, {"running": [{"model": m, "state": "ready"} for m in models]}


def test_holders_with_cards_and_persistence(config_file: Path) -> None:
    with FakeService({("GET", "/running"): answers(running("main-a", "describer"))}).running() as fake:
        holders = LlamaSwap("ls", fake.url, config_file).holders(GPUS)
    assert [(h.item, h.gpu_uuids, h.persistent) for h in holders] == [
        ("main-a", frozenset({"GPU-aaaa0000", "GPU-bbbb1111"}), False),
        ("describer", frozenset({"GPU-bbbb1111"}), True),
    ]


def test_a_vllm_model_claims_its_share_of_each_card(config_file: Path) -> None:
    with FakeService({("GET", "/running"): answers(running("main-vllm", "main-a"))}).running() as fake:
        holders = LlamaSwap("ls", fake.url, config_file).holders(GPUS)
    assert [(h.item, h.memory_fraction) for h in holders] == [("main-vllm", 0.9), ("main-a", None)]


def test_running_model_missing_from_config_is_an_error(config_file: Path) -> None:
    service = FakeService({("GET", "/running"): answers(running("ghost"))})
    with service.running() as fake, pytest.raises(ConsumerError, match="ghost"):
        LlamaSwap("ls", fake.url, config_file).holders(GPUS)


def test_model_cards(config_file: Path) -> None:
    server = LlamaSwap("ls", "http://unused", config_file)
    assert server.model_gpu_uuids("main-vlm-b", GPUS) == {"GPU-bbbb1111"}
    assert server.model_gpu_uuids("main-all", GPUS) == {gpu.uuid for gpu in GPUS}
    with pytest.raises(UnknownModel):
        server.model_gpu_uuids("ghost", GPUS)
    with pytest.raises(GpuSpecError):
        server.model_gpu_uuids("broken", GPUS)


def test_eviction_rules(config_file: Path) -> None:
    server = LlamaSwap("ls", "http://unused", config_file, no_eviction_models=["*-vlm-*"])
    assert server.evicts_persistent("main-a")
    assert not server.evicts_persistent("main-vlm-b")
    assert not server.evicts_persistent("describer")


def test_release_checks_that_the_model_is_gone(config_file: Path) -> None:
    routes = {
        # llama-swap answers an unload with the plain text "OK", not JSON.
        ("POST", "/api/models/unload/describer"): answers((200, b"OK")),
        ("POST", "/api/models/unload/embedder"): answers((200, b"OK")),
        ("GET", "/running"): answers(running("embedder")),
    }
    holders = [Holder("ls", "describer", frozenset()), Holder("ls", "embedder", frozenset())]
    with FakeService(routes).running() as fake:
        actions = LlamaSwap("ls", fake.url, config_file).release(holders, reason="r", deadline=time.monotonic() + 60)
    assert [(a.item, a.outcome) for a in actions] == [("describer", "released"), ("embedder", "error")]
