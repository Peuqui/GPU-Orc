"""The measurement store and the one-time import of AIfred's calibration."""

import json
from pathlib import Path

import pytest

from gpu_orc.measurements import (
    CardMeasurement,
    MeasurementError,
    Measurements,
    ProfileMeasurement,
    VisionMeasurement,
    import_aifred,
    load,
    save,
)

UUIDS = ["GPU-aaaa0000", "GPU-bbbb1111"]


def calibration(measured_at: str, free: list[int] | None, context: int | None = 262144) -> dict[str, object]:
    entry: dict[str, object] = {"max_context": context, "measured_at": measured_at}
    if free is not None:
        entry.update(vram_per_gpu="49152,32768", gpu_uuids=UUIDS, remaining_free_mb=free)
    return entry


def aifred_data(tmp_path: Path) -> Path:
    (tmp_path / "model_vram_cache.json").write_text(json.dumps({
        "small": {"backend": "llamacpp", "llamacpp_calibrations": [
            calibration("2026-07-01T10:00:00", [9000, 32000]),
            calibration("2026-07-12T23:02:40", [4987, 32181]),   # the newest one counts
        ]},
        "small-tts-xtts": {"backend": "llamacpp", "llamacpp_calibrations": [calibration("2026-07-12T23:03:00", None)]},
        "no-context": {"backend": "llamacpp", "llamacpp_calibrations": [
            calibration("2026-07-12T23:04:00", [4000, 32000], context=None)]},
        "vllm-model": {"backend": "vllm"},
    }))
    (tmp_path / "vlm_vram_cache.json").write_text(json.dumps({
        "Qwen3VL-4B-Instruct-Q8_0": {"num_ctx": 24576, "peak_mb": 7396, "measured_at": "2026-10-09T23:31:09",
                                     "source": "stress_prewarm"},
    }))
    return tmp_path


def test_import_takes_the_newest_card_measurement_of_each_profile_and_the_vision_peaks(tmp_path: Path) -> None:
    imported = import_aifred(aifred_data(tmp_path))
    # A copy without free memory per card, a calibration without its context and vLLM entries bring none.
    assert set(imported.profiles) == {"small"}
    small = imported.profiles["small"]
    assert small.cards == {"GPU-aaaa0000": CardMeasurement(49152, 4987), "GPU-bbbb1111": CardMeasurement(32768, 32181)}
    assert small.context == 262144
    assert small.measured_at == "2026-07-12T23:02:40" and small.source.endswith("model_vram_cache.json")
    assert imported.vision["Qwen3VL-4B-Instruct-Q8_0"] == VisionMeasurement(
        7396, 24576, "2026-10-09T23:31:09", str(tmp_path / "vlm_vram_cache.json"))


def test_lists_of_different_length_are_an_error(tmp_path: Path) -> None:
    data = aifred_data(tmp_path)
    cache = json.loads((data / "model_vram_cache.json").read_text())
    cache["small"]["llamacpp_calibrations"][1]["remaining_free_mb"] = [4987]
    (data / "model_vram_cache.json").write_text(json.dumps(cache))
    with pytest.raises(MeasurementError, match="differ in length"):
        import_aifred(data)


def test_a_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(MeasurementError, match="model_vram_cache.json"):
        import_aifred(tmp_path)


def test_store_round_trip_and_an_empty_start(tmp_path: Path) -> None:
    store = tmp_path / "state" / "measurements.json"
    assert load(store) == Measurements()
    measurements = Measurements(
        profiles={"small": ProfileMeasurement({"GPU-aaaa0000": CardMeasurement(49152, 4987)}, 262144, "2026-07-12", "x")},
        vision={"vl": VisionMeasurement(7396, None, "2026-10-09", "y")},
    )
    save(store, measurements)
    assert load(store) == measurements
    assert not store.with_suffix(".tmp").exists()


def test_a_store_in_an_unknown_form_is_an_error(tmp_path: Path) -> None:
    store = tmp_path / "measurements.json"
    store.write_text('{"profiles": []}')
    with pytest.raises(MeasurementError):
        load(store)
