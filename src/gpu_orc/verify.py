"""Checking that a release really freed the GPU instead of believing the service's answer.

A released item passes when (a) its service no longer reports it and (b) the GPU memory went down:
by at least half of what the service reported for it, or, where it reported nothing, by at least
``min_freed_mib`` over its cards together (at most what was in use there before: a container that had no
CUDA context yet leaves nothing to go down). The driver often returns memory a second or two after the
process ended, so the check is repeated until it passes or the time is up.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

from .model import Action, Holder

RELEASED = ("released", "released_forced")
# What a holder that reported its memory must give back of it, at least.
REPORTED_SHARE = 0.5


@dataclass(frozen=True)
class VerifySettings:
    timeout_s: float = 30.0
    min_freed_mib: int = 256
    poll_s: float = 0.5


DEFAULT_VERIFY = VerifySettings()


@dataclass(frozen=True)
class StillHeld:
    """What the services report after the release: (consumer, item) pairs, and the consumers that
    could not report (their items cannot be checked)."""

    items: frozenset[tuple[str, str]]
    unreachable: Mapping[str, str]


def verify(
    actions: Sequence[Action],
    plan: Mapping[str, Sequence[Holder]],
    before: Mapping[str, int],
    measure: Callable[[], dict[str, int]],
    still_held: Callable[[], StillHeld],
    settings: VerifySettings,
    deadline: float,
) -> tuple[list[Action], dict[str, int]]:
    """The actions with each failed release turned into ``not_freed``, and the last memory measured
    per GPU UUID (MiB used)."""
    holders = {(name, holder.item): holder for name, consumer_holders in plan.items() for holder in consumer_holders}
    released = [holders[(a.consumer, a.item)] for a in actions if a.outcome in RELEASED and (a.consumer, a.item) in holders]
    end = min(time.monotonic() + settings.timeout_s, deadline)
    while True:
        after = measure()
        if not released:
            return list(actions), after
        failures = _failures(released, before, after, still_held(), settings.min_freed_mib)
        if not failures or time.monotonic() >= end:
            break
        time.sleep(min(settings.poll_s, max(0.0, end - time.monotonic())))
    checked = []
    for action in actions:
        detail = failures.get((action.consumer, action.item))
        checked.append(replace(action, outcome="not_freed", detail=detail) if detail else action)
    return checked, after


def _failures(
    released: Sequence[Holder],
    before: Mapping[str, int],
    after: Mapping[str, int],
    held: StillHeld,
    min_freed_mib: int,
) -> dict[tuple[str, str], str]:
    drop = {uuid: before[uuid] - after[uuid] for uuid in before if uuid in after}
    # Holders that reported memory share the expectation of their card.
    expected: dict[str, float] = {}
    for holder in released:
        for uuid, mib in holder.memory_mib.items():
            expected[uuid] = expected.get(uuid, 0.0) + REPORTED_SHARE * mib
    failures = {}
    for holder in released:
        status = _status(holder, held)
        memory = _memory(holder, before, after, drop, expected, min_freed_mib)
        if status or memory:
            failures[(holder.consumer, holder.item)] = (
                f"status: {status or 'no longer reported'}; memory: {memory or 'went down'}"
            )
    return failures


def _status(holder: Holder, held: StillHeld) -> str:
    if holder.consumer in held.unreachable:
        return f"cannot check, {held.unreachable[holder.consumer]}"
    if (holder.consumer, holder.item) in held.items:
        return "still reported"
    return ""


def _memory(
    holder: Holder,
    before: Mapping[str, int],
    after: Mapping[str, int],
    drop: Mapping[str, int],
    expected: Mapping[str, float],
    min_freed_mib: int,
) -> str:
    if holder.memory_mib:
        short = [uuid for uuid in holder.memory_mib if drop.get(uuid, 0) < expected[uuid]]
        return "; ".join(
            f"{uuid} {before.get(uuid)} → {after.get(uuid)} MiB, expected −{round(expected[uuid])}" for uuid in short
        )
    if not holder.gpu_uuids:
        return ""
    freed = sum(max(0, drop.get(uuid, 0)) for uuid in holder.gpu_uuids)
    # A card cannot give back more than was in use (found live: a TTS start withdrawn before its CUDA context).
    needed = min(min_freed_mib, sum(before.get(uuid, 0) for uuid in holder.gpu_uuids))
    if freed >= needed:
        return ""
    return f"−{freed} MiB over {len(holder.gpu_uuids)} card(s), expected at least −{needed}"
