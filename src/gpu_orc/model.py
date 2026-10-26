"""What a consumer holds, what GPU-Orc did about it, and the consumer interfaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from .gpus import Gpu

if TYPE_CHECKING:
    from .measurements import Measurements
    from .profiles import Profile

Outcome = Literal[
    "released", "released_forced", "not_freed", "refused_busy", "timeout", "kept", "error", "would_release"
]


@dataclass(frozen=True)
class Holder:
    """Something a consumer keeps on one or more GPUs, as its own service reports it."""

    consumer: str
    item: str
    gpu_uuids: frozenset[str]
    # MiB per GPU UUID, only where the service reports its memory.
    memory_mib: Mapping[str, int] = field(default_factory=dict)
    busy: bool | None = None
    persistent: bool | None = None
    # What a side model needs on each of its cards, as its service reports it (the larger of this and
    # its memory is its claim); a model of the model server has the claim of its profile instead.
    need_mib: int | None = None


NeedSource = Literal["measured", "configured", "estimate", "unknown"]


@dataclass(frozen=True)
class Need:
    """What one item of a consumer needs on a card, as its service reports it (``GET /gpu/needs``) or, for
    a vision model of the model server, as GPU-Orc measured it. ``default`` = the item the service loads on
    a GPU when a request names none."""

    consumer: str
    item: str
    need_mib: int | None
    source: NeedSource
    default: bool = False


@dataclass(frozen=True)
class Action:
    consumer: str
    item: str
    outcome: Outcome
    detail: str = ""
    waited_s: float | None = None


class ConsumerError(Exception):
    """A consumer's service did not answer, or not as its protocol says."""


class UnknownModel(KeyError):
    """The model server's config has no entry for the model."""


class Consumer(Protocol):
    name: str

    def holders(self, gpus: Sequence[Gpu]) -> list[Holder]:
        """What the service holds right now. Raises ConsumerError."""
        ...

    def release(self, holders: Sequence[Holder], *, reason: str, deadline: float) -> list[Action]:
        """Ask the service to let go of ``holders``; one action per holder. ``deadline`` is a
        time.monotonic() value no waiting may pass. Raises ConsumerError."""
        ...


@runtime_checkable
class SideModelServer(Consumer, Protocol):
    """A service of side models (the suites): it says what its items need and asks GPU-Orc where to start one."""

    def needs(self) -> list[Need]:
        """What each item needs on a card, running or not. Raises ConsumerError."""
        ...

    def release_idle(self, holders: Sequence[Holder], *, reason: str, deadline: float) -> list[Action]:
        """Ask once, without force and without waiting, to let go of ``holders``; a busy one stays
        (``refused_busy``). Raises ConsumerError."""
        ...


@runtime_checkable
class ModelServer(Consumer, Protocol):
    def is_running(self, model: str) -> bool:
        """Whether the model is loaded or loading. Raises ConsumerError."""
        ...

    def model_gpu_uuids(self, model: str, gpus: Sequence[Gpu]) -> frozenset[str]:
        """The GPUs the model would use. Raises UnknownModel or GpuSpecError."""
        ...

    def evicts_persistent(self, model: str) -> bool:
        """Whether loading the model evicts this server's persistent models from shared GPUs."""
        ...

    def profiles(self, gpus: Sequence[Gpu], measurements: Measurements) -> list[Profile]:
        """Every profile of the server's config. Raises ConsumerError."""
        ...
