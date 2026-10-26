"""What each card has left for a side model, and the card the placement rule picks.

A claim is what an item may take on a card, whether it holds that much yet or not:

- a model of the model server: the claim of its profile (see :mod:`gpu_orc.profiles`), the whole card while
  that is unknown;
- a side model (an engine or worker of a suite): the larger of the memory its service reports and the need it
  reports (the memory still grows while it starts, and peaks while it computes), the whole card while the need
  is unknown;
- room GPU-Orc promised (a reservation) and no holder accounts for yet: its need.

Memory in use that no holder accounts for (another program) counts on top. The room of a card is its memory
minus all that. A side model fits where the room minus the headroom covers its need; of those cards one nobody
uses comes first (side models spread over the free cards instead of piling onto one), then the one with the
most room, then the lowest index. Where none fits, idle side models of a service with lower priority may have
to give way.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace

from .gpus import Gpu
from .model import Holder


@dataclass(frozen=True)
class PlaceSettings:
    headroom_mib: int  # stays free on a card on top of a side model's need
    reserve_s: float  # how long promised room counts without a holder that accounts for it
    priority: tuple[str, ...]  # side-model services, the first gives way last


DEFAULT_PLACE = PlaceSettings(headroom_mib=512, reserve_s=60.0, priority=())


@dataclass(frozen=True)
class Claim:
    consumer: str
    item: str
    mib: int
    in_use_mib: int  # what it holds now; its claim where nobody reports that
    busy: bool | None


@dataclass(frozen=True)
class CardRoom:
    gpu: Gpu
    claims: tuple[Claim, ...]

    @property
    def room_mib(self) -> int:
        return room_mib(self.gpu, self.claims)

    @property
    def unused(self) -> bool:
        """Nothing claims the card and nothing holds memory on it."""
        return not self.claims and self.gpu.memory_used_mib == 0


@dataclass(frozen=True)
class Eviction:
    """The card where the item fits once the idle side models in ``claims`` let go."""

    room: CardRoom
    claims: tuple[Claim, ...]


def room_mib(gpu: Gpu, claims: Sequence[Claim]) -> int:
    stranger = max(0, gpu.memory_used_mib - sum(claim.in_use_mib for claim in claims))
    return max(0, gpu.memory_total_mib - sum(claim.mib for claim in claims) - stranger)


def side_model_claim(holder: Holder, gpu: Gpu) -> Claim:
    memory = holder.memory_mib.get(gpu.uuid)
    mib = gpu.memory_total_mib if holder.need_mib is None else max(holder.need_mib, memory or 0)
    return Claim(holder.consumer, holder.item, mib, mib if memory is None else memory, holder.busy)


def model_claim(holder: Holder, gpu: Gpu, profile_claim: Mapping[str, int] | None) -> Claim:
    """A model of the model server, which reports no memory: its claim counts as in use."""
    mib = profile_claim[gpu.uuid] if profile_claim is not None and gpu.uuid in profile_claim else gpu.memory_total_mib
    return Claim(holder.consumer, holder.item, mib, mib, holder.busy)


def with_reservations(claims: Sequence[Claim], reservations: Sequence[Reservation]) -> tuple[Claim, ...]:
    """The claims of one card with the reservations on it: a reservation of an item that holds the card
    already raises its claim to the reserved need, any other counts as a claim of its own."""
    merged = list(claims)
    for reservation in reservations:
        key = (reservation.consumer, reservation.item)
        index = next((i for i, claim in enumerate(merged) if (claim.consumer, claim.item) == key), None)
        if index is None:
            merged.append(Claim(reservation.consumer, reservation.item, reservation.need_mib, 0, None))
        else:
            merged[index] = replace(merged[index], mib=max(merged[index].mib, reservation.need_mib))
    return tuple(merged)


def choose(rooms: Sequence[CardRoom], need_mib: int, headroom_mib: int) -> CardRoom | None:
    fitting = [room for room in rooms if room.room_mib - headroom_mib >= need_mib]
    return min(fitting, key=lambda room: (not room.unused, -room.room_mib, room.gpu.index), default=None)


def plan_eviction(
    rooms: Sequence[CardRoom], need_mib: int, headroom_mib: int, rank: Callable[[Claim], int | None],
) -> Eviction | None:
    """The card where letting go of the fewest idle side models makes the item fit; of those the most room
    afterwards, then the lowest index. ``rank``: higher gives way sooner, ``None`` never. On each card the
    highest rank goes first, of the same rank the largest claim."""
    options: list[tuple[int, int, int, Eviction]] = []
    for room in rooms:
        ranked = [(rank_of, claim) for claim in room.claims
                  if claim.busy is False and (rank_of := rank(claim)) is not None]
        evicted: list[Claim] = []
        for _rank, claim in sorted(ranked, key=lambda pair: (-pair[0], -pair[1].mib)):
            evicted.append(claim)
            gpu = replace(room.gpu, memory_used_mib=room.gpu.memory_used_mib - sum(e.in_use_mib for e in evicted))
            after = room_mib(gpu, [kept for kept in room.claims if kept not in evicted])
            if after - headroom_mib >= need_mib:
                options.append((len(evicted), -after, room.gpu.index, Eviction(room, tuple(evicted))))
                break
    return min(options, key=lambda option: option[:3])[3] if options else None


@dataclass(frozen=True)
class Reservation:
    consumer: str
    item: str
    card: str  # GPU UUID
    need_mib: int
    expires: float  # time.monotonic()


class Reservations:
    """Room GPU-Orc promised and no holder may account for yet: a side model between the answer of ``/place``
    and its start, a model between ``/free`` and its load. Each runs out at its ``expires``. The lock is held
    only while a decision is computed and entered (``deciding``), never while a service is asked."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._lock = threading.RLock()
        self._clock = clock
        self._entries: dict[tuple[str, str, str], Reservation] = {}

    @contextmanager
    def deciding(self) -> Iterator[None]:
        with self._lock:
            yield

    def active(self) -> list[Reservation]:
        with self._lock:
            now = self._clock()
            for key in [key for key, reservation in self._entries.items() if reservation.expires <= now]:
                del self._entries[key]
            return list(self._entries.values())

    def put(self, consumer: str, item: str, needs: Mapping[str, int], lasting_s: float) -> None:
        """Promise ``needs`` (MiB per GPU UUID) to the item, in place of what it was promised before."""
        with self._lock:
            self.drop(consumer, item)
            expires = self._clock() + lasting_s
            for card, need in needs.items():
                self._entries[(consumer, item, card)] = Reservation(consumer, item, card, need, expires)

    def drop(self, consumer: str, item: str) -> None:
        with self._lock:
            for key in [key for key in self._entries if key[:2] == (consumer, item)]:
                del self._entries[key]

    def left_s(self, reservation: Reservation) -> float:
        return max(0.0, round(reservation.expires - self._clock(), 1))
