"""The GPUs of this machine as nvidia-smi sees them, and which of them a CUDA_VISIBLE_DEVICES names."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

NVIDIA_SMI_TIMEOUT_S = 10


@dataclass(frozen=True)
class Gpu:
    index: int  # nvidia-smi index = PCI bus order, what CUDA_DEVICE_ORDER=PCI_BUS_ID numbers
    uuid: str
    name: str
    memory_total_mib: int
    memory_used_mib: int
    compute_cap: float  # CUDA compute capability (7.0 = Volta, 7.5 = Turing)


class GpuSpecError(ValueError):
    """A GPU named in a request or in a model server's config that is not on this machine."""


def query_gpus() -> list[Gpu]:
    """All GPUs with their memory. nvidia-smi reports memory per card also under WSL2, where it lists
    no processes; GPU-Orc never needs the process list."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=NVIDIA_SMI_TIMEOUT_S,
        check=True,
    )
    return parse_gpu_rows(result.stdout)


def parse_gpu_rows(text: str) -> list[Gpu]:
    gpus = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        # The name may contain a comma; the other fields are fixed at both ends.
        index, uuid, total, used, compute_cap = parts[0], parts[1], parts[-3], parts[-2], parts[-1]
        name = ", ".join(parts[2:-3])
        gpus.append(Gpu(int(index), uuid, name, int(total), int(used), float(compute_cap)))
    return sorted(gpus, key=lambda gpu: gpu.index)


def resolve_gpu(spec: int | str, gpus: Sequence[Gpu]) -> Gpu:
    """A GPU named by its nvidia-smi index or by its UUID (also a unique prefix of it)."""
    if isinstance(spec, bool) or not isinstance(spec, int | str):
        raise GpuSpecError(f"{spec!r} is neither a GPU index nor a GPU UUID")
    if isinstance(spec, int):
        for gpu in gpus:
            if gpu.index == spec:
                return gpu
        raise GpuSpecError(f"GPU index {spec} is not on this machine")
    matches = [gpu for gpu in gpus if spec.startswith("GPU-") and gpu.uuid.startswith(spec)]
    if len(matches) != 1:
        raise GpuSpecError(f"{spec!r} names no single GPU of this machine")
    return matches[0]


def visible_gpu_uuids(env: Mapping[str, str], gpus: Sequence[Gpu]) -> frozenset[str]:
    """The GPUs a process started with ``env`` may use.

    Without CUDA_VISIBLE_DEVICES it sees every GPU; ``CUDA_VISIBLE_DEVICES=`` means none. Entries name
    cards by UUID (also a unique prefix) or by index under CUDA_DEVICE_ORDER=PCI_BUS_ID. An index
    without that order is CUDA's FASTEST_FIRST numbering, which nvidia-smi cannot map: that raises
    instead of guessing.
    """
    if "CUDA_VISIBLE_DEVICES" not in env:
        return frozenset(gpu.uuid for gpu in gpus)
    uuids: set[str] = set()
    for token in (part.strip() for part in env["CUDA_VISIBLE_DEVICES"].split(",")):
        if not token:
            continue
        if token.startswith("GPU-"):
            uuids.add(resolve_gpu(token, gpus).uuid)
        elif token.isdigit():
            if env.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
                raise GpuSpecError(
                    f"CUDA_VISIBLE_DEVICES index {token} without CUDA_DEVICE_ORDER=PCI_BUS_ID "
                    "cannot be mapped to a GPU"
                )
            uuids.add(resolve_gpu(int(token), gpus).uuid)
        else:
            raise GpuSpecError(f"CUDA_VISIBLE_DEVICES entry {token!r} is not a GPU")
    return frozenset(uuids)
