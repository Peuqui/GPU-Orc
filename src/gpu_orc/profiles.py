"""The model server's profiles as GPU-Orc plans with them.

Every entry of llama-swap's config is a profile: its cards, its backend, its context and KV blocks, its
variant (which side models the calibration left room for) and what it takes of each of its cards. That
last number is measured where a calibration recorded what stayed free after loading at the profile's
context (a variant the calibration wrote as a copy of its base, entry for entry, takes its base's
measurement), and for a vision describer where the burn-in of its base model ran at that context; for a vLLM
entry without a measurement it is its ``--gpu-memory-utilization`` share of each card (vLLM allocates part
of it only while it computes); an entry without cards takes nothing; otherwise it is unknown.

Variant names follow the calibration: ``<base>[-tts-<engine>][-vlm-<key>][-speed]`` for the main models,
``<base>-visiond[-gpu<N>]`` for the vision describers.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .gpus import Gpu, GpuSpecError, visible_gpu_uuids
from .measurements import Measurements, ProfileMeasurement

Backend = Literal["vllm", "llama.cpp", "other"]
ClaimSource = Literal["measured", "fraction", "no_cards", "unknown"]

_GPU_MEMORY_UTILIZATION = re.compile(r"--gpu-memory-utilization[=\s]+([0-9]*\.?[0-9]+)")
_MAX_MODEL_LEN = re.compile(r"--max-model-len[=\s]+(\d+)")
_LLAMA_CONTEXT = re.compile(r"(?:^|\s)(?:-c|--ctx-size)[=\s]+(\d+)")
_KV_BLOCKS = re.compile(r"--num-gpu-blocks-override[=\s]+(\d+)")
_DESCRIBER = re.compile(r"^(?P<base>.+)-visiond(?:-gpu(?P<gpu>\d+))?$")
_SPEED = re.compile(r"^(?P<rest>.+)-speed$")
_VLM = re.compile(r"^(?P<rest>.+)-vlm-(?P<key>[A-Za-z0-9.]+)$")
_TTS = re.compile(r"^(?P<rest>.+)-tts-(?P<engine>[A-Za-z0-9]+)$")


@dataclass(frozen=True)
class Variant:
    base: str
    tts: str | None = None  # the TTS engine the profile leaves room for
    vlm: str | None = None  # the vision model the profile leaves room for
    speed: bool = False
    describer: bool = False  # a vision describer profile (-visiond)
    describer_gpu: int | None = None  # its pinned card (-visiond-gpu<N>)


@dataclass(frozen=True)
class Profile:
    name: str
    backend: Backend
    cards: tuple[str, ...]  # GPU UUIDs in nvidia-smi index order
    persistent: bool
    context: int | None
    memory_fraction: float | None
    kv_blocks: int | None
    variant: Variant
    claim_mib: Mapping[str, int] | None  # what it takes per card; None = unknown
    claim_source: ClaimSource
    measured_at: str | None
    measured_as: str | None = None  # the profile with the same entry whose measurement it takes
    problem: str | None = None  # why the cards could not be read


def parse_variant(name: str) -> Variant:
    describer = _DESCRIBER.match(name)
    if describer:
        gpu = describer.group("gpu")
        return Variant(base=describer.group("base"), describer=True, describer_gpu=int(gpu) if gpu else None)
    rest, speed, vlm, tts = name, False, None, None
    if match := _SPEED.match(rest):
        rest, speed = match.group("rest"), True
    if match := _VLM.match(rest):
        rest, vlm = match.group("rest"), match.group("key")
    if match := _TTS.match(rest):
        rest, tts = match.group("rest"), match.group("engine")
    return Variant(base=rest, tts=tts, vlm=vlm, speed=speed)


def read_profiles(
    entries: Mapping[str, Mapping[str, Any]],
    envs: Mapping[str, Mapping[str, str]],
    persistent: frozenset[str],
    gpus: Sequence[Gpu],
    measurements: Measurements,
) -> list[Profile]:
    """One profile per entry of the model server's config, in the config's order."""
    by_uuid = {gpu.uuid: gpu for gpu in gpus}
    profiles = []
    for name, entry in entries.items():
        cmd = str(entry.get("cmd") or "")
        problem = None
        try:
            cards = tuple(sorted(visible_gpu_uuids(envs[name], gpus), key=lambda uuid: by_uuid[uuid].index))
        except GpuSpecError as error:
            cards, problem = (), str(error)
        fraction = memory_fraction(cmd)
        context = _number(_MAX_MODEL_LEN, cmd, int) or _number(_LLAMA_CONTEXT, cmd, int)
        variant = parse_variant(name)
        claim, source, measured_at, measured_as = (
            (None, "unknown", None, None) if problem is not None
            else _claim(variant, context, cards, fraction, by_uuid, _measurement_of(name, entries, measurements),
                        measurements)
        )
        profiles.append(Profile(
            name=name,
            backend=_backend(cmd),
            cards=cards,
            persistent=name in persistent,
            context=context,
            memory_fraction=fraction,
            kv_blocks=_number(_KV_BLOCKS, cmd, int),
            variant=variant,
            claim_mib=claim,
            claim_source=source,
            measured_at=measured_at,
            measured_as=measured_as,
            problem=problem,
        ))
    return profiles


def memory_fraction(cmd: str) -> float | None:
    """vLLM's ``--gpu-memory-utilization`` in an entry's ``cmd``; ``None`` for an entry without it."""
    return _number(_GPU_MEMORY_UTILIZATION, cmd, float)


def _measurement_of(
    name: str, entries: Mapping[str, Mapping[str, Any]], measurements: Measurements,
) -> tuple[ProfileMeasurement, str | None] | None:
    """The profile's own calibration, else that of a profile with the very same entry (the calibration
    writes a variant whose room lies on a card outside the model as a copy of its base), with that
    profile's name."""
    if name in measurements.profiles:
        return measurements.profiles[name], None
    for other, measured in measurements.profiles.items():
        if entries.get(other) == entries[name]:
            return measured, other
    return None


def _claim(
    variant: Variant,
    context: int | None,
    cards: tuple[str, ...],
    fraction: float | None,
    by_uuid: Mapping[str, Gpu],
    profile_measurement: tuple[ProfileMeasurement, str | None] | None,
    measurements: Measurements,
) -> tuple[dict[str, int] | None, ClaimSource, str | None, str | None]:
    if not cards:
        return {}, "no_cards", None, None  # CUDA_VISIBLE_DEVICES= : the entry runs on the CPU
    if profile_measurement is not None:
        measured, measured_as = profile_measurement
        if measured.context == context and all(uuid in measured.cards for uuid in cards):
            claim = {uuid: measured.cards[uuid].total_mib - measured.cards[uuid].free_mib for uuid in cards}
            return claim, "measured", measured.measured_at, measured_as
    burn_in = measurements.vision.get(variant.base) if variant.describer else None
    if burn_in is not None and burn_in.context == context:
        return {uuid: burn_in.need_mib for uuid in cards}, "measured", burn_in.measured_at, None
    if fraction is not None:
        return {uuid: round(fraction * by_uuid[uuid].memory_total_mib) for uuid in cards}, "fraction", None, None
    return None, "unknown", None, None


def _backend(cmd: str) -> Backend:
    if "vllm" in cmd:
        return "vllm"
    if "llama-server" in cmd:
        return "llama.cpp"
    return "other"


def _number[T: (int, float)](pattern: re.Pattern[str], cmd: str, kind: type[T]) -> T | None:
    match = pattern.search(cmd)
    return kind(match.group(1)) if match else None
