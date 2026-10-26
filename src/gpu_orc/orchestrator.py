"""Who holds what, freeing GPUs (plan from the consumers' own reports, ask them in parallel) and placing a side
model on a card (see :mod:`gpu_orc.claims`)."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Literal

from .claims import (
    DEFAULT_PLACE,
    CardRoom,
    Claim,
    Eviction,
    PlaceSettings,
    Reservation,
    Reservations,
    choose,
    model_claim,
    plan_eviction,
    side_model_claim,
    with_reservations,
)
from .gate import FifoGate
from .gpus import Gpu, GpuSpecError, query_gpus, resolve_gpu
from .measurements import Measurements
from .model import Action, Consumer, ConsumerError, Holder, ModelServer, Need, SideModelServer, UnknownModel
from .profiles import Profile
from .verify import DEFAULT_VERIFY, StillHeld, VerifySettings, verify

# How much longer than the overall limit a caller's HTTP timeout should be (answer, nvidia-smi).
CALLER_MARGIN_S = 30.0


class RequestError(ValueError):
    """The request names something that does not exist or is malformed (HTTP 400)."""


class NotFound(LookupError):
    """The request names a model server that is not configured (HTTP 404)."""


class ModelServerUnreachable(RuntimeError):
    """The model server could not say whether the model runs (HTTP 502)."""


@dataclass(frozen=True)
class FreeRequest:
    reason: str
    gpus: tuple[int | str, ...] | Literal["all"] | None = None
    model: str | None = None
    model_server: str | None = None
    keep: tuple[str, ...] = ()
    dry_run: bool = False


@dataclass(frozen=True)
class PlaceRequest:
    consumer: str
    item: str
    need_mib: int
    reason: str
    card: int | str | None = None  # only this card (index or UUID)
    min_compute_cap: float | None = None


@dataclass(frozen=True)
class _Report:
    holders: dict[str, list[Holder]]
    errors: list[dict[str, str]]


class Orchestrator:
    def __init__(
        self,
        consumers: Sequence[Consumer],
        free_timeout_s: float,
        query: Callable[[], list[Gpu]] = query_gpus,
        verify_settings: VerifySettings = DEFAULT_VERIFY,
        measurements: Callable[[], Measurements] = Measurements,
        vision_models: Mapping[str, Sequence[str]] | None = None,
        place_settings: PlaceSettings = DEFAULT_PLACE,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._consumers = {consumer.name: consumer for consumer in consumers}
        # Read on every call: an import or a measurement counts at once.
        self._measurements = measurements
        self._vision_models = vision_models or {}
        self._free_timeout_s = free_timeout_s
        self._query = query
        self._verify = verify_settings
        self._gate = FifoGate()
        self._place = place_settings
        self._reservations = Reservations(clock)

    @property
    def free_timeout_s(self) -> float:
        return self._free_timeout_s

    @property
    def caller_timeout_s(self) -> float:
        return self._free_timeout_s + CALLER_MARGIN_S

    def status(self) -> dict[str, Any]:
        gpus = self._query()
        state = self._gate.state()
        return {
            "queue": {"running": state.running, "waiting": state.waiting, "running_since_s": state.running_since_s},
            **self._status_body(gpus, self._collect(gpus), [gpu.uuid for gpu in gpus]),
        }

    def profiles(self) -> dict[str, Any]:
        """Every profile of every model server, with what it takes of its cards."""
        gpus = self._query()
        measurements = self._measurements()
        profiles: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for name, consumer in self._consumers.items():
            if not isinstance(consumer, ModelServer):
                continue
            try:
                profiles.extend(_profile_json(name, profile) for profile in consumer.profiles(gpus, measurements))
            except ConsumerError as error:
                errors.append({"consumer": name, "detail": str(error)})
        return {"profiles": profiles, "errors": errors}

    def needs(self) -> dict[str, Any]:
        """What the side models need on a card: the suites' items as they report them, the vision models
        of the model servers as measured."""
        reporters = {name: c for name, c in self._consumers.items() if isinstance(c, SideModelServer)}
        needs: list[Need] = []
        errors: list[dict[str, str]] = []
        if reporters:
            with ThreadPoolExecutor(max_workers=len(reporters)) as pool:
                futures = {name: pool.submit(reporter.needs) for name, reporter in reporters.items()}
            for name, future in futures.items():
                try:
                    needs.extend(future.result())
                except ConsumerError as error:
                    errors.append({"consumer": name, "detail": str(error)})
        vision = self._measurements().vision
        for server, models in self._vision_models.items():
            for model in models:
                measured = vision.get(model)
                needs.append(Need(server, model, measured.need_mib, "measured") if measured is not None
                             else Need(server, model, None, "unknown"))
        return {"needs": [_need_json(need) for need in needs], "errors": errors}

    def free(self, request: FreeRequest) -> dict[str, Any]:
        started = time.monotonic()
        deadline = started + self._free_timeout_s
        gpus = self._query()
        server = self._model_server(request.model_server) if request.model is not None else None
        targets = self._targets(request, server, gpus)
        self._check_keep(request.keep)
        if server is not None and self._running(server, request.model):
            return self._already_running(gpus, targets, started)
        with self._gate.enter(deadline) as entered:
            # A request ahead in the queue may have loaded the model meanwhile.
            if entered and server is not None and self._running(server, request.model):
                return self._already_running(gpus, targets, started)
            loading = (server, request.model) if entered and not request.dry_run and server is not None and \
                request.model is not None else None
            if loading is not None:
                # The cards stay the model's while they are cleared: a side model placed meanwhile would take its room.
                needs = self._model_needs(*loading, gpus)
                self._reservations.put(loading[0].name, loading[1], needs, deadline - time.monotonic() + self._place.reserve_s)
            report = self._collect(gpus)
            plan, kept = self._plan(request, server, targets, report.holders)
            before = self._memory()
            memory_after: dict[str, int] | None = None
            if not entered:
                actions = [Action(name, h.item, "timeout", "overall limit reached in the queue")
                           for name, holders in plan.items() for h in holders]
            elif request.dry_run:
                actions = [Action(name, h.item, "would_release") for name, holders in plan.items() for h in holders]
            else:
                actions, memory_after = verify(
                    self._release(plan, request.reason, deadline),
                    plan,
                    before,
                    self._memory,
                    lambda: self._still_held(gpus),
                    self._verify,
                    deadline,
                )
            actions = self._in_config_order(actions + kept + self._error_actions(report, request.keep))
            after = report if request.dry_run else self._collect(gpus)
            remaining, _ = self._plan(request, server, targets, after.holders)
        failed = any(action.outcome in ("not_freed", "error") for action in actions)
        unknown = _unkept_errors(after, request.keep)
        result = {
            "gpus": [{"index": gpu.index, "uuid": gpu.uuid} for gpu in gpus if gpu.uuid in targets],
            "already_running": False,
            "duration_s": round(time.monotonic() - started, 1),
            "actions": [_action_json(action) for action in actions],
            "memory": [
                {
                    "index": gpu.index,
                    "uuid": gpu.uuid,
                    "before_mib": before.get(gpu.uuid),
                    "after_mib": None if memory_after is None else memory_after.get(gpu.uuid),
                }
                for gpu in gpus
                if gpu.uuid in targets
            ],
            "free": not any(remaining.values()) and not unknown and not failed,
            "status": self._status_body(self._query(), after, targets),
        }
        if loading is not None:
            self._reservations.put(loading[0].name, loading[1], needs, self._place.reserve_s)  # the caller loads it next
        _log(request, result)
        return result

    def place(self, request: PlaceRequest) -> dict[str, Any]:
        """The card a side model may start on, its need promised there for ``reserve_s``: a card it fits on as
        things are, else one where idle side models of a service with lower priority let go first; no card, and
        why, when there is none."""
        started = time.monotonic()
        self._side_model_server(request.consumer)
        gpus = self._query()
        allowed = self._allowed_cards(request, gpus)
        report = self._collect(gpus)
        with self._reservations.deciding():
            card, eviction, why = self._choose_card(request, gpus, report, allowed)
        actions: list[Action] = []
        if eviction is not None:
            actions = self._evict(request, eviction, report)
            card, why = self._after_eviction(request, eviction.room.gpu.uuid, actions)
        result = {
            "card": None if card is None else {"index": card.index, "uuid": card.uuid},
            "why": why,
            "reserved_s": None if card is None else self._place.reserve_s,
            "actions": [_action_json(action) for action in actions],
            "duration_s": round(time.monotonic() - started, 1),
        }
        _log_place(request, result)
        return result

    # --- where a side model goes --------------------------------------------------------------

    def _side_model_server(self, name: str) -> SideModelServer:
        consumer = self._consumers.get(name)
        if not isinstance(consumer, SideModelServer):
            known = sorted(n for n, c in self._consumers.items() if isinstance(c, SideModelServer))
            raise NotFound(f"no side-model service {name!r}, configured: {known}")
        return consumer

    @staticmethod
    def _allowed_cards(request: PlaceRequest, gpus: Sequence[Gpu]) -> frozenset[str]:
        cards = list(gpus)
        if request.card is not None:
            try:
                cards = [resolve_gpu(request.card, gpus)]
            except GpuSpecError as error:
                raise RequestError(str(error)) from error
        if request.min_compute_cap is not None:
            cards = [gpu for gpu in cards if gpu.compute_cap >= request.min_compute_cap]
        return frozenset(gpu.uuid for gpu in cards)

    def _choose_card(
        self, request: PlaceRequest, gpus: Sequence[Gpu], report: _Report, allowed: frozenset[str],
    ) -> tuple[Gpu | None, Eviction | None, str | None]:
        """Under the lock: the card, promised; or the card where idle side models must let go first (promised
        already, so nobody takes the room they leave); or why there is none."""
        rooms, errors = self._rooms(gpus, report, without=(request.consumer, request.item))
        if errors:
            return None, None, f"cannot say what is free: {_names(errors)}"
        candidates = [room for room in rooms if room.gpu.uuid in allowed]
        if not candidates:
            return None, None, "no card of this machine meets the request"
        headroom = self._place.headroom_mib
        chosen = choose(candidates, request.need_mib, headroom)
        if chosen is not None:
            self._promise(request, chosen.gpu)
            return chosen.gpu, None, None
        eviction = plan_eviction(candidates, request.need_mib, headroom, self._rank(request.consumer))
        if eviction is None:
            best = max(candidates, key=lambda room: room.room_mib)
            return None, None, (f"needs {request.need_mib} MiB and {headroom} MiB headroom, the most room is "
                                f"{best.room_mib} MiB on GPU {best.gpu.index}")
        self._promise(request, eviction.room.gpu)
        return None, eviction, None

    def _promise(self, request: PlaceRequest, gpu: Gpu) -> None:
        self._reservations.put(request.consumer, request.item, {gpu.uuid: request.need_mib}, self._place.reserve_s)

    def _rank(self, consumer: str) -> Callable[[Claim], int | None]:
        """Who gives way to ``consumer``: the side models of a service after it in ``priority``, the later
        the sooner."""
        priority = self._place.priority
        if consumer not in priority:
            return lambda claim: None
        own = priority.index(consumer)

        def rank(claim: Claim) -> int | None:
            if claim.consumer not in priority or priority.index(claim.consumer) <= own:
                return None
            return priority.index(claim.consumer)

        return rank

    def _evict(self, request: PlaceRequest, eviction: Eviction, report: _Report) -> list[Action]:
        reason = f"room for {request.consumer}:{request.item}: {request.reason}"
        deadline = time.monotonic() + self._free_timeout_s
        wanted = {(claim.consumer, claim.item) for claim in eviction.claims}
        actions: list[Action] = []
        for name, holders in report.holders.items():
            evicted = [holder for holder in holders if (name, holder.item) in wanted]
            service = self._consumers[name]
            if not evicted or not isinstance(service, SideModelServer):
                continue
            try:
                actions.extend(service.release_idle(evicted, reason=reason, deadline=deadline))
            except ConsumerError as error:
                actions.extend(Action(name, holder.item, "error", str(error)) for holder in evicted)
        return actions

    def _after_eviction(self, request: PlaceRequest, uuid: str, actions: Sequence[Action]) -> tuple[Gpu | None, str | None]:
        """The card once the others let go and its memory came back (within the verify limit), promised anew;
        else no card and why."""
        refused = [action for action in actions if action.outcome != "released"]
        if refused:
            self._reservations.drop(request.consumer, request.item)
            return None, "did not let go: " + ", ".join(f"{a.consumer}:{a.item} ({a.outcome})" for a in refused)
        limit = time.monotonic() + self._verify.timeout_s
        while True:
            gpus = self._query()
            report = self._collect(gpus)
            with self._reservations.deciding():
                rooms, errors = self._rooms(gpus, report, without=(request.consumer, request.item))
                room = next(room for room in rooms if room.gpu.uuid == uuid)
                if not errors and room.room_mib - self._place.headroom_mib >= request.need_mib:
                    self._promise(request, room.gpu)
                    return room.gpu, None
                if time.monotonic() >= limit:
                    self._reservations.drop(request.consumer, request.item)
                    why = f"cannot say what is free: {_names(errors)}" if errors else \
                        f"GPU {room.gpu.index} has {room.room_mib} MiB room after the release"
                    return None, why
            time.sleep(self._verify.poll_s)

    # --- what each card has left --------------------------------------------------------------

    def _rooms(
        self, gpus: Sequence[Gpu], report: _Report, without: tuple[str, str] | None = None,
    ) -> tuple[list[CardRoom], list[dict[str, str]]]:
        """Every card with the claims on it, and the consumers whose claims are not known. ``without``: a
        consumer's item that is being replaced; neither its holder, nor the memory that holder has in use,
        nor its promise counts."""
        profile_claims, errors = self._profile_claims(gpus, report)
        reservations = [r for r in self._reservations.active() if (r.consumer, r.item) != without]
        rooms = []
        for gpu in gpus:
            claims = [self._claim(holder, gpu, profile_claims)
                      for holders in report.holders.values() for holder in holders if gpu.uuid in holder.gpu_uuids]
            replaced = sum(claim.in_use_mib for claim in claims if (claim.consumer, claim.item) == without)
            kept = [claim for claim in claims if (claim.consumer, claim.item) != without]
            on_card = [reservation for reservation in reservations if reservation.card == gpu.uuid]
            card = replace(gpu, memory_used_mib=max(0, gpu.memory_used_mib - replaced))
            rooms.append(CardRoom(card, with_reservations(kept, on_card)))
        return rooms, report.errors + errors

    def _profile_claims(
        self, gpus: Sequence[Gpu], report: _Report,
    ) -> tuple[dict[tuple[str, str], Mapping[str, int] | None], list[dict[str, str]]]:
        """The claim of each running model of the model servers by (server, model); ``None`` = unknown."""
        claims: dict[tuple[str, str], Mapping[str, int] | None] = {}
        errors: list[dict[str, str]] = []
        for name, holders in report.holders.items():
            server = self._consumers[name]
            if not holders or not isinstance(server, ModelServer):
                continue
            try:
                profiles = {profile.name: profile.claim_mib for profile in server.profiles(gpus, self._measurements())}
            except ConsumerError as error:
                errors.append({"consumer": name, "detail": str(error)})
                continue
            claims.update({(name, holder.item): profiles.get(holder.item) for holder in holders})
        return claims, errors

    def _claim(self, holder: Holder, gpu: Gpu, profile_claims: Mapping[tuple[str, str], Mapping[str, int] | None]) -> Claim:
        if isinstance(self._consumers[holder.consumer], ModelServer):
            return model_claim(holder, gpu, profile_claims.get((holder.consumer, holder.item)))
        return side_model_claim(holder, gpu)

    def _model_needs(self, server: ModelServer, model: str, gpus: Sequence[Gpu]) -> dict[str, int]:
        """What the model's profile claims per card, the whole card where that is unknown."""
        try:
            profile = next(profile for profile in server.profiles(gpus, self._measurements()) if profile.name == model)
        except ConsumerError as error:
            raise ModelServerUnreachable(str(error)) from error
        totals = {gpu.uuid: gpu.memory_total_mib for gpu in gpus}
        claim = profile.claim_mib or {}
        return {uuid: claim.get(uuid, totals[uuid]) for uuid in profile.cards}

    # --- what to free -------------------------------------------------------------------------

    def _model_server(self, name: str | None) -> ModelServer:
        servers = {n: c for n, c in self._consumers.items() if isinstance(c, ModelServer)}
        if name is None:
            if len(servers) != 1:
                raise RequestError(f"model_server is required, configured: {sorted(servers)}")
            return next(iter(servers.values()))
        if name not in servers:
            raise NotFound(f"no model server {name!r}, configured: {sorted(servers)}")
        return servers[name]

    def _targets(self, request: FreeRequest, server: ModelServer | None, gpus: Sequence[Gpu]) -> frozenset[str]:
        if server is not None and request.model is not None:
            try:
                return server.model_gpu_uuids(request.model, gpus)
            except UnknownModel as error:
                raise RequestError(f"model {request.model!r} is not in the config of {server.name}") from error
            except GpuSpecError as error:
                raise RequestError(f"{server.name}: {request.model}: {error}") from error
        if request.gpus == "all":
            return frozenset(gpu.uuid for gpu in gpus)
        try:
            return frozenset(resolve_gpu(spec, gpus).uuid for spec in request.gpus or ())
        except GpuSpecError as error:
            raise RequestError(str(error)) from error

    def _check_keep(self, keep: Sequence[str]) -> None:
        unknown = [entry for entry in keep if entry.split(":", 1)[0] not in self._consumers]
        if unknown:
            raise RequestError(f"keep names unknown consumers: {unknown}")

    @staticmethod
    def _running(server: ModelServer, model: str | None) -> bool:
        try:
            return model is not None and server.is_running(model)
        except ConsumerError as error:
            raise ModelServerUnreachable(str(error)) from error

    def _plan(
        self,
        request: FreeRequest,
        server: ModelServer | None,
        targets: frozenset[str],
        holders: Mapping[str, Sequence[Holder]],
    ) -> tuple[dict[str, list[Holder]], list[Action]]:
        """What each consumer has to let go of, and the actions for what ``keep`` protects.

        The ``model`` form leaves the target server's other models to the server, which swaps them
        out itself when it loads; only its persistent ones stay loaded, so they must go, unless the
        model is configured to evict nothing.
        """
        evicts = server.evicts_persistent(request.model) if server is not None and request.model else True
        plan: dict[str, list[Holder]] = {}
        kept: list[Action] = []
        for name, consumer_holders in holders.items():
            for holder in consumer_holders:
                if not holder.gpu_uuids & targets:
                    continue
                if server is not None and name == server.name and not (holder.persistent and evicts):
                    continue
                if name in request.keep or f"{name}:{holder.item}" in request.keep:
                    kept.append(Action(name, holder.item, "kept"))
                    continue
                plan.setdefault(name, []).append(holder)
        return plan, kept

    # --- asking the consumers -----------------------------------------------------------------

    def _collect(self, gpus: Sequence[Gpu]) -> _Report:
        holders: dict[str, list[Holder]] = {}
        errors: list[dict[str, str]] = []
        with ThreadPoolExecutor(max_workers=len(self._consumers)) as pool:
            futures = {name: pool.submit(consumer.holders, gpus) for name, consumer in self._consumers.items()}
        for name, future in futures.items():
            try:
                holders[name] = future.result()
            except ConsumerError as error:
                errors.append({"consumer": name, "detail": str(error)})
        return _Report(holders, errors)

    def _memory(self) -> dict[str, int]:
        return {gpu.uuid: gpu.memory_used_mib for gpu in self._query()}

    def _still_held(self, gpus: Sequence[Gpu]) -> StillHeld:
        report = self._collect(gpus)
        items = frozenset((name, holder.item) for name, holders in report.holders.items() for holder in holders)
        return StillHeld(items, {error["consumer"]: error["detail"] for error in report.errors})

    def _release(self, plan: Mapping[str, Sequence[Holder]], reason: str, deadline: float) -> list[Action]:
        if not plan:
            return []
        actions: list[Action] = []
        with ThreadPoolExecutor(max_workers=len(plan)) as pool:
            futures = {
                name: pool.submit(self._consumers[name].release, holders, reason=reason, deadline=deadline)
                for name, holders in plan.items()
            }
        for name, future in futures.items():
            try:
                actions.extend(future.result())
            except ConsumerError as error:
                actions.extend(Action(name, holder.item, "error", str(error)) for holder in plan[name])
        return actions

    @staticmethod
    def _error_actions(report: _Report, keep: Sequence[str]) -> list[Action]:
        """A consumer that could not say what it holds may sit on the cards: say so per request."""
        return [Action(error["consumer"], "", "error", error["detail"]) for error in _unkept_errors(report, keep)]

    def _in_config_order(self, actions: list[Action]) -> list[Action]:
        order = {name: index for index, name in enumerate(self._consumers)}
        return sorted(actions, key=lambda action: order[action.consumer])

    # --- answers ------------------------------------------------------------------------------

    def _already_running(self, gpus: Sequence[Gpu], targets: frozenset[str], started: float) -> dict[str, Any]:
        return {
            "gpus": [{"index": gpu.index, "uuid": gpu.uuid} for gpu in gpus if gpu.uuid in targets],
            "already_running": True,
            "duration_s": round(time.monotonic() - started, 1),
            "actions": [],
            "free": True,
            "status": None,
        }

    def _status_body(self, gpus: Sequence[Gpu], report: _Report, uuids: Sequence[str] | frozenset[str]) -> dict[str, Any]:
        rooms, errors = self._rooms(gpus, report)
        reservations = self._reservations.active()
        return {
            "gpus": [self._gpu_json(room, report.holders, reservations, complete=not errors)
                     for room in rooms if room.gpu.uuid in uuids],
            "errors": errors,
        }

    def _gpu_json(
        self, room: CardRoom, holders: Mapping[str, Sequence[Holder]], reservations: Sequence[Reservation], *,
        complete: bool,
    ) -> dict[str, Any]:
        """One card of ``/status``; ``complete``: every claim is known (else nobody can say what is available)."""
        gpu = room.gpu
        on_card = [holder for consumer_holders in holders.values() for holder in consumer_holders
                   if gpu.uuid in holder.gpu_uuids]
        reported = [holder.memory_mib.get(gpu.uuid) for holder in on_card]
        if all(mib is not None for mib in reported):
            unattributed: int | None = max(0, gpu.memory_used_mib - sum(mib for mib in reported if mib is not None))
        else:
            unattributed = None
        claims = {(claim.consumer, claim.item): claim.mib for claim in room.claims}
        return {
            "index": gpu.index,
            "uuid": gpu.uuid,
            "name": gpu.name,
            "compute_cap": gpu.compute_cap,
            "memory_total_mib": gpu.memory_total_mib,
            "memory_used_mib": gpu.memory_used_mib,
            "holders": [
                {
                    "consumer": holder.consumer,
                    "item": holder.item,
                    "memory_mib": holder.memory_mib.get(gpu.uuid),
                    "need_mib": holder.need_mib,
                    "claim_mib": claims[(holder.consumer, holder.item)] if complete else None,
                    "busy": holder.busy,
                    "persistent": holder.persistent,
                }
                for holder in on_card
            ],
            "reservations": [
                {"consumer": r.consumer, "item": r.item, "need_mib": r.need_mib, "left_s": self._reservations.left_s(r)}
                for r in reservations if r.card == gpu.uuid
            ],
            "unattributed_mib": unattributed,
            "available_mib": room.room_mib if complete else None,
        }


def _unkept_errors(report: _Report, keep: Sequence[str]) -> list[dict[str, str]]:
    """The consumers that could not report, except those the request keeps as a whole: what they hold
    stays either way (calibration stops the model server, then has the rest cleared)."""
    return [error for error in report.errors if error["consumer"] not in keep]


def _profile_json(server: str, profile: Profile) -> dict[str, Any]:
    variant = profile.variant
    return {
        "name": profile.name,
        "model_server": server,
        "backend": profile.backend,
        "cards": list(profile.cards),
        "persistent": profile.persistent,
        "context": profile.context,
        "memory_fraction": profile.memory_fraction,
        "kv_blocks": profile.kv_blocks,
        "variant": {
            "base": variant.base, "tts": variant.tts, "vlm": variant.vlm, "speed": variant.speed,
            "describer": variant.describer, "describer_gpu": variant.describer_gpu,
        },
        "claim_mib": dict(profile.claim_mib) if profile.claim_mib is not None else None,
        "claim_source": profile.claim_source,
        "measured_at": profile.measured_at,
        "measured_as": profile.measured_as,
        "problem": profile.problem,
    }


def _need_json(need: Need) -> dict[str, Any]:
    return {"consumer": need.consumer, "item": need.item, "need_mib": need.need_mib, "source": need.source,
            "default": need.default}


def _action_json(action: Action) -> dict[str, Any]:
    body: dict[str, Any] = {"consumer": action.consumer, "item": action.item, "outcome": action.outcome}
    if action.detail:
        body["detail"] = action.detail
    if action.waited_s is not None:
        body["waited_s"] = action.waited_s
    return body


def _names(errors: Sequence[Mapping[str, str]]) -> str:
    return ", ".join(f"{error['consumer']} ({error['detail']})" for error in errors)


def _log_place(request: PlaceRequest, result: Mapping[str, Any]) -> None:
    card = f"GPU {result['card']['index']}" if result["card"] is not None else f"no card: {result['why']}"
    evicted = "".join(f", {a['consumer']}:{a['item']}={a['outcome']}" for a in result["actions"])
    print(f"[place] {request.consumer}:{request.item} {request.need_mib} MiB — {request.reason} — {card}{evicted} "
          f"in {result['duration_s']} s", flush=True)


def _log(request: FreeRequest, result: Mapping[str, Any]) -> None:
    target = f"model {request.model}" if request.model else f"gpus {request.gpus}"
    outcomes = ", ".join(f"{a['consumer']}:{a['item']}={a['outcome']}" for a in result["actions"]) or "nothing to do"
    mode = " (dry run)" if request.dry_run else ""
    print(f"[free{mode}] {target} — {request.reason} — {outcomes} — free={result['free']} "
          f"in {result['duration_s']} s", flush=True)
