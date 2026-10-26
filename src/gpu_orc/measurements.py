"""What GPU-Orc knows from measurements: what a model-server profile left free on each card after it
loaded (calibration), and what a vision model needs on a card (burn-in).

Kept in ``<state_dir>/measurements.json``. The values of an existing calibration are taken over once with
``gpu-orc import-aifred`` (AIfred's ``data/model_vram_cache.json`` and ``data/vlm_vram_cache.json``); later
GPU-Orc measures them itself.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

FILE_NAME = "measurements.json"


class MeasurementError(ValueError):
    """A measurement file that cannot be read, or one in an unknown form."""


@dataclass(frozen=True)
class CardMeasurement:
    total_mib: int
    free_mib: int  # what was free on the card with the profile loaded


@dataclass(frozen=True)
class ProfileMeasurement:
    cards: Mapping[str, CardMeasurement]  # by GPU UUID
    context: int  # the context it was loaded with; it counts only for a profile with the same
    measured_at: str
    source: str


@dataclass(frozen=True)
class VisionMeasurement:
    need_mib: int
    context: int | None
    measured_at: str
    source: str


@dataclass(frozen=True)
class Measurements:
    profiles: Mapping[str, ProfileMeasurement] = field(default_factory=dict)
    vision: Mapping[str, VisionMeasurement] = field(default_factory=dict)


def load(path: Path) -> Measurements:
    """The stored measurements; none while the file does not exist."""
    if not path.exists():
        return Measurements()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Measurements(
            profiles={
                name: ProfileMeasurement(
                    cards={uuid: CardMeasurement(**card) for uuid, card in entry["cards"].items()},
                    context=entry["context"],
                    measured_at=entry["measured_at"],
                    source=entry["source"],
                )
                for name, entry in data["profiles"].items()
            },
            vision={name: VisionMeasurement(**entry) for name, entry in data["vision"].items()},
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise MeasurementError(f"cannot read {path}: {error!r}") from error


def save(path: Path, measurements: Measurements) -> None:
    """Write atomically: a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(asdict(measurements), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def import_aifred(data_dir: Path) -> Measurements:
    """The measurements of AIfred's calibration: for each llama.cpp profile its latest calibration that
    recorded what stayed free per card and at which context, and the burn-in peak of each vision model."""
    model_cache = _read_json(data_dir / "model_vram_cache.json")
    vision_cache = _read_json(data_dir / "vlm_vram_cache.json")
    source_models = str(data_dir / "model_vram_cache.json")
    profiles: dict[str, ProfileMeasurement] = {}
    for name, entry in model_cache.items():
        measured = _latest_card_measurement(entry.get("llamacpp_calibrations") or [], source_models)
        if measured is not None:
            profiles[name] = measured
    source_vision = str(data_dir / "vlm_vram_cache.json")
    vision = {
        name: VisionMeasurement(
            need_mib=int(entry["peak_mb"]),
            context=int(entry["num_ctx"]) if entry.get("num_ctx") is not None else None,
            measured_at=str(entry["measured_at"]),
            source=source_vision,
        )
        for name, entry in vision_cache.items()
    }
    return Measurements(profiles=profiles, vision=vision)


def _latest_card_measurement(calibrations: list[Mapping[str, Any]], source: str) -> ProfileMeasurement | None:
    """The newest calibration with the free memory per card and its context; ``None`` when none recorded
    them (a copied variant, or a calibration from before that was recorded)."""
    complete = [
        c for c in calibrations
        if c.get("remaining_free_mb") and c.get("gpu_uuids") and c.get("vram_per_gpu") and c.get("max_context")
    ]
    if not complete:
        return None
    latest = max(complete, key=lambda calibration: str(calibration["measured_at"]))
    totals = [int(total) for total in str(latest["vram_per_gpu"]).split(",")]
    uuids, free = list(latest["gpu_uuids"]), [int(mib) for mib in latest["remaining_free_mb"]]
    if not len(uuids) == len(free) == len(totals):
        raise MeasurementError(f"{source}: gpu_uuids, remaining_free_mb and vram_per_gpu differ in length")
    return ProfileMeasurement(
        cards={uuid: CardMeasurement(total_mib=total, free_mib=mib) for uuid, total, mib in zip(uuids, totals, free)},
        context=int(latest["max_context"]),
        measured_at=str(latest["measured_at"]),
        source=source,
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise MeasurementError(f"cannot read {path}: {error}") from error
    if not isinstance(data, dict):
        raise MeasurementError(f"{path}: an object is expected")
    return data
