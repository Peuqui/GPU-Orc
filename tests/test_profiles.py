"""Profiles: variant names, backend, cards, context, KV blocks and what a profile takes of its cards."""

from typing import Any

import pytest
from fakes import GPUS

from gpu_orc.measurements import CardMeasurement, Measurements, ProfileMeasurement, VisionMeasurement
from gpu_orc.profiles import Variant, memory_fraction, parse_variant, read_profiles

A, B, C = (gpu.uuid for gpu in GPUS)  # 49152, 32768 and 49152 MiB


@pytest.mark.parametrize("name, variant", [
    ("Qwen3VL-4B-Instruct-Q8_0", Variant("Qwen3VL-4B-Instruct-Q8_0")),
    ("Qwen3VL-4B-Instruct-Q8_0-tts-qwen3local", Variant("Qwen3VL-4B-Instruct-Q8_0", tts="qwen3local")),
    ("Flash-Next-TP2-vllm-vlm-qwen3vl4b", Variant("Flash-Next-TP2-vllm", vlm="qwen3vl4b")),
    ("Flash-Next-TP2-vllm-tts-xtts-vlm-qwen3vl4b-speed",
     Variant("Flash-Next-TP2-vllm", tts="xtts", vlm="qwen3vl4b", speed=True)),
    ("Model-27B-speed", Variant("Model-27B", speed=True)),
    ("Qwen3VL-4B-Instruct-Q8_0-visiond", Variant("Qwen3VL-4B-Instruct-Q8_0", describer=True)),
    ("Qwen3VL-4B-Instruct-Q8_0-visiond-gpu4", Variant("Qwen3VL-4B-Instruct-Q8_0", describer=True, describer_gpu=4)),
])
def test_variant_names_follow_the_calibration(name: str, variant: Variant) -> None:
    assert parse_variant(name) == variant


@pytest.mark.parametrize("cmd, fraction", [
    ("vllm serve --gpu-memory-utilization 0.95 --port 1", 0.95),
    ("vllm serve --gpu-memory-utilization=.9", 0.9),
    ("llama-server -c 4096", None),
])
def test_memory_fraction(cmd: str, fraction: float | None) -> None:
    assert memory_fraction(cmd) == fraction


ENTRIES: dict[str, dict[str, Any]] = {
    "main-vllm": {"cmd": "python -m vllm.entrypoints.openai.api_server --gpu-memory-utilization 0.9 "
                         "--max-model-len 307200 --num-gpu-blocks-override 3500"},
    "small": {"cmd": "/x/llama-server --port ${PORT} -c 262144 -ctk q8_0"},
    "small-tts-xtts": {"cmd": "/x/llama-server --port ${PORT} -c 131072"},
    "broken": {"cmd": "x"},
    "embed": {"cmd": "/x/llama-server --port ${PORT} -c 8192 -ngl 0"},
    "small-visiond-gpu1": {"cmd": "/x/llama-server --port ${PORT} -c 24576"},
}
ENVS = {
    "main-vllm": {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "2,0"},
    "small": {"CUDA_VISIBLE_DEVICES": A},
    "small-tts-xtts": {"CUDA_VISIBLE_DEVICES": A},
    "broken": {"CUDA_VISIBLE_DEVICES": "3"},
    "embed": {"CUDA_VISIBLE_DEVICES": ""},
    "small-visiond-gpu1": {"CUDA_VISIBLE_DEVICES": B},
}
MEASURED = Measurements(
    profiles={"small": ProfileMeasurement(
        cards={A: CardMeasurement(49152, 4987), B: CardMeasurement(32768, 32181)}, context=262144,
        measured_at="2026-07-12T23:02:40", source="model_vram_cache.json")},
    vision={"small": VisionMeasurement(7396, 24576, "2026-10-09T23:31:09", "vlm_vram_cache.json")},
)


def test_profiles_with_their_cards_settings_and_what_they_take() -> None:
    main, small, small_tts, broken, embed, describer = read_profiles(
        ENTRIES, ENVS, frozenset({"small"}), GPUS, MEASURED)

    assert (main.backend, main.cards, main.context, main.kv_blocks, main.memory_fraction) == (
        "vllm", (A, C), 307200, 3500, 0.9)  # cards in nvidia-smi order, whatever order the config names them
    assert (main.claim_mib, main.claim_source, main.measured_at) == ({A: 44237, C: 44237}, "fraction", None)

    assert (small.backend, small.cards, small.context, small.kv_blocks, small.persistent) == (
        "llama.cpp", (A,), 262144, None, True)
    # Measured: total minus what stayed free, on the profile's own cards only.
    assert (small.claim_mib, small.claim_source, small.measured_at) == ({A: 44165}, "measured", "2026-07-12T23:02:40")

    assert (small_tts.variant, small_tts.context) == (Variant("small", tts="xtts"), 131072)
    # Its own settings and no measurement of its own: unknown.
    assert (small_tts.claim_mib, small_tts.claim_source, small_tts.measured_as) == (None, "unknown", None)

    assert (broken.backend, broken.cards, broken.claim_mib, broken.claim_source) == ("other", (), None, "unknown")
    assert broken.problem is not None and "PCI_BUS_ID" in broken.problem

    # CUDA_VISIBLE_DEVICES= : runs on the CPU and takes nothing of any card.
    assert (embed.cards, embed.claim_mib, embed.claim_source, embed.problem) == ((), {}, "no_cards", None)

    # A describer takes what the burn-in of its base model needed at the same context.
    assert describer.variant == Variant("small", describer=True, describer_gpu=1)
    assert (describer.claim_mib, describer.claim_source, describer.measured_at) == (
        {B: 7396}, "measured", "2026-10-09T23:31:09")


def test_a_copy_of_a_measured_profile_takes_its_measurement() -> None:
    # The calibration writes a variant whose room lies on a card outside the model as a copy of its base.
    entries = {"small": ENTRIES["small"], "small-tts-qwen3local": dict(ENTRIES["small"])}
    envs = {"small": ENVS["small"], "small-tts-qwen3local": ENVS["small"]}
    small, copy = read_profiles(entries, envs, frozenset(), GPUS, MEASURED)
    assert (small.claim_mib, small.measured_as) == ({A: 44165}, None)
    assert (copy.claim_mib, copy.claim_source, copy.measured_at, copy.measured_as) == (
        {A: 44165}, "measured", "2026-07-12T23:02:40", "small")


def test_a_measurement_at_another_context_does_not_count() -> None:
    entries = {"small": {"cmd": "/x/llama-server -c 131072"}, "small-visiond": {"cmd": "/x/llama-server -c 8192"}}
    envs = {"small": {"CUDA_VISIBLE_DEVICES": A}, "small-visiond": {"CUDA_VISIBLE_DEVICES": B}}
    small, describer = read_profiles(entries, envs, frozenset(), GPUS, MEASURED)
    assert (small.claim_mib, small.claim_source) == (None, "unknown")
    assert (describer.claim_mib, describer.claim_source) == (None, "unknown")


def test_a_measurement_that_misses_a_card_of_the_profile_does_not_count() -> None:
    partial = Measurements(profiles={"main-vllm": ProfileMeasurement(
        cards={A: CardMeasurement(49152, 1000)}, context=307200, measured_at="2026-10-25", source="x")})
    (main,) = read_profiles({"main-vllm": ENTRIES["main-vllm"]}, ENVS, frozenset(), GPUS, partial)
    assert main.claim_source == "fraction"
