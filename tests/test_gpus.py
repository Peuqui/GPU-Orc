import pytest
from fakes import GPUS

from gpu_orc.gpus import GpuSpecError, parse_gpu_rows, resolve_gpu, visible_gpu_uuids

ALL = frozenset(gpu.uuid for gpu in GPUS)


def test_parse_rows_sorted_by_index_and_name_with_comma() -> None:
    text = "1, GPU-b, Tesla V100, 32GB, 32768, 10, 7.0\n0, GPU-a, Quadro RTX 8000, 49152, 2000, 7.5\n"
    gpus = parse_gpu_rows(text)
    assert [gpu.index for gpu in gpus] == [0, 1]
    assert gpus[1].name == "Tesla V100, 32GB"
    assert (gpus[1].memory_total_mib, gpus[1].memory_used_mib, gpus[1].compute_cap) == (32768, 10, 7.0)


def test_resolve_by_index_uuid_and_unique_prefix() -> None:
    assert resolve_gpu(1, GPUS).uuid == "GPU-bbbb1111"
    assert resolve_gpu("GPU-cccc2222", GPUS).index == 2
    assert resolve_gpu("GPU-aaaa", GPUS).index == 0


@pytest.mark.parametrize("spec", [7, "GPU-", "GPU-zzzz", "bbbb1111", True, 1.0])
def test_resolve_rejects_what_names_no_single_gpu(spec: int | str) -> None:
    # True and 1.0 arrive like this from JSON although the signature excludes them.
    with pytest.raises(GpuSpecError):
        resolve_gpu(spec, GPUS)


def test_without_cuda_visible_devices_every_gpu() -> None:
    assert visible_gpu_uuids({}, GPUS) == ALL


def test_empty_cuda_visible_devices_no_gpu() -> None:
    assert visible_gpu_uuids({"CUDA_VISIBLE_DEVICES": ""}, GPUS) == frozenset()


def test_uuids_and_pci_indices() -> None:
    env = {"CUDA_VISIBLE_DEVICES": "GPU-aaaa, 2", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    assert visible_gpu_uuids(env, GPUS) == {"GPU-aaaa0000", "GPU-cccc2222"}


@pytest.mark.parametrize("env", [
    {"CUDA_VISIBLE_DEVICES": "1"},  # FASTEST_FIRST numbering cannot be mapped
    {"CUDA_VISIBLE_DEVICES": "5", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
    {"CUDA_VISIBLE_DEVICES": "MIG-xyz"},
])
def test_unmappable_entries_raise(env: dict[str, str]) -> None:
    with pytest.raises(GpuSpecError):
        visible_gpu_uuids(env, GPUS)
