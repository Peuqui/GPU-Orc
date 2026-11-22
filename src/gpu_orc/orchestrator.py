"""Who holds what, and freeing GPUs: plan from the consumers' own reports, ask them in parallel."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal, NoReturn

from .gate import FifoGate
from .gpus import Gpu, GpuSpecError, query_gpus, resolve_gpu
from .hold import ACTIVE, REQUESTED, Handover, Hold, HoldStore
from .model import Action, Consumer, ConsumerError, Holder, ModelServer, UnknownModel
from .verify import DEFAULT_VERIFY, StillHeld, VerifySettings, verify
from .work import WorkStore

# How much longer than the overall limit a caller's HTTP timeout should be (answer, nvidia-smi).
CALLER_MARGIN_S = 30.0
# How long a model that was cleared for counts as loading after /free ended, until its server says so.
DEFAULT_LOAD_GRACE_S = 15.0
# How often the hold worker looks again whether a requested hold may clear the cards.
DEFAULT_HOLD_CHECK_INTERVAL_S = 30.0
# How long the cards stay for the person who asked an owner to end its hold, after the owner let go.
DEFAULT_HANDOVER_GRACE_S = 120.0


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
        load_grace_s: float = DEFAULT_LOAD_GRACE_S,
        hold_check_interval_s: float = DEFAULT_HOLD_CHECK_INTERVAL_S,
        handover_grace_s: float = DEFAULT_HANDOVER_GRACE_S,
        *,
        hold: HoldStore,
        work: WorkStore,
    ) -> None:
        self._consumers = {consumer.name: consumer for consumer in consumers}
        self._hold = hold
        self._hold_worker: threading.Thread | None = None
        self._hold_wake = threading.Event()
        self._hold_check_interval_s = hold_check_interval_s
        self._handover_grace_s = handover_grace_s
        self._cards_note: str | None = None  # what the worker last reported about a model server it cannot ask
        self._work = work
        self._load_grace_s = load_grace_s
        self._announced: dict[str, float] = {}  # model -> time.monotonic() until it counts as loading
        self._announced_lock = threading.Lock()
        self._free_timeout_s = free_timeout_s
        self._query = query
        self._verify = verify_settings
        self._gate = FifoGate()

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
            **self._hold_light(),
            "handover": self._handover_json(),
            "work": self._work.active(),
            **self._status_body(gpus, self._collect(gpus), [gpu.uuid for gpu in gpus]),
        }

    def hold(self, owner: str, reason: str, model: str | None, model_server: str | None) -> dict[str, Any]:
        """``owner`` asks to keep the cards (hold.py); ``model`` is the model of the model server that the owner
        needs and that stays loaded, None for an owner that needs none (a game). A new hold is only requested;
        the hold worker clears the cards and makes it active once no main model is loaded or loading any more
        (callers wait with ``hold_and_wait``). Asking again as the same owner answers the hold as it is; another
        owner is refused (HoldConflict) while any hold exists. Raises RequestError, NotFound for a model or
        model server that is not configured."""
        server = self._model_server(model_server) if model is not None else None
        if server is not None:
            self._targets(FreeRequest(reason=reason, model=model, model_server=server.name), server, self._query())
        hold = self._hold.request(owner, reason, model, None if server is None else server.name)
        self._hold_wake.set()
        return {"hold": hold.as_json()}

    def grant_hold(self, owner: str, by: str) -> dict[str, Any]:
        """The person ``by`` grants the requested hold of ``owner`` (kept in the hold file): once no work of
        others runs, the worker clears the cards for exactly this hold without asking whether they are free.
        Raises NotFound without a hold, HoldConflict when the hold belongs to somebody else; an active hold is
        answered as it is."""
        hold = self._hold.grant(owner, by)
        if hold is None:
            raise NotFound("there is no hold")
        if hold.granted_by == by and hold.state == REQUESTED:
            self._hold_wake.set()
            print(f"[hold] GRANTED by {by!r} for {hold.owner!r} ({hold.reason!r}, since {hold.since})",
                  file=sys.stderr, flush=True)
        return {"hold": hold.as_json()}

    def request_hold_end(self, owner: str, by: str) -> dict[str, Any]:
        """The person ``by`` asks ``owner`` to end the hold. A requested hold is withdrawn at once (nothing is
        on the cards); an active one only gets the note ``end_requested_by`` and stays, GPU-Orc tells nobody:
        the owner asks for the note between its steps and lets go itself. Raises NotFound without a hold,
        HoldConflict when the hold belongs to somebody else."""
        hold = self._hold.request_end(owner, by)
        if hold is None:
            raise NotFound("there is no hold")
        if hold.state == REQUESTED:
            print(f"[hold] WITHDRAWN on request of {by!r}: {hold.owner!r} ({hold.reason!r}, since {hold.since})",
                  file=sys.stderr, flush=True)
            return {"hold": None}
        if hold.end_requested_by == by:
            print(f"[hold] END REQUESTED by {by!r} for {hold.owner!r} ({hold.reason!r}, since {hold.since})",
                  file=sys.stderr, flush=True)
        return {"hold": hold.as_json()}

    def start_hold_worker(self) -> None:
        """Start the one thread that turns a requested hold into an active one, for the life of the
        service. A hold found at start (requested before a restart) waits for its moment the same way."""
        if self._hold_worker is not None:
            raise RuntimeError("the hold worker runs already")
        self._hold_worker = threading.Thread(target=self._run_hold_worker, name="hold-worker", daemon=True)
        self._hold_worker.start()

    def _run_hold_worker(self) -> None:
        """Only this thread clears the cards for a hold, so there are never two clearings, and a hold
        never waits for a thread that is about to end. It sleeps until the next look (``hold_check_interval_s``,
        or the end of a handover when that comes sooner) or until ``hold``, ``grant_hold``, ``release_hold`` or
        ``end_work`` wakes it. The wake is cleared before each look, so a wake during
        the look or the clearing is not lost.
        Should it fail (writing the hold file, a bug), every hold would stay requested for ever while the
        service answers on: it ends the service loudly instead, systemd starts it again (Restart=on-failure)."""
        try:
            while True:
                self._hold_wake.clear()
                self._hold_wake.wait(self._settle_hold())
        except Exception:  # noqa: BLE001 — the only thread that activates holds died, the service must not live on
            print(f"[hold] the hold worker failed, GPU-Orc ends:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
            _end_process(1)

    def _settle_hold(self) -> float | None:
        """One look at the hold: clear the cards for a requested hold whose cards are free, or that is granted
        and no work of others runs,
        and make that very hold active with the result, however it ended (nobody retries behind the owner's
        back), so whoever sees ``active`` finds the cards cleared or the reason why not. While a handover stands,
        no hold turns active, except a granted one (the person decides). Returns how long to
        sleep: the check interval while the hold waits (no longer than a handover stands), the rest of a handover
        when no hold waits, None until somebody wakes the worker. The hold is read before the handover:
        ``release`` sets the handover under the lock of the hold, so a hold requested after a release never
        meets a handover that is not there yet."""
        hold = self._hold.get()
        handover = self._hold.handover()
        if hold is None or hold.state != REQUESTED:
            return self._handover_left(handover, busy=[])
        if hold.granted_by is None:
            busy = self._busy_models(hold)
            left = self._handover_left(handover, busy)
            if left is not None:
                return min(self._hold_check_interval_s, left)
            if busy != []:  # a main model, or a model server that cannot say
                return self._hold_check_interval_s
        else:
            if handover is not None:
                self._end_handover(handover, "granted")
            if self._work.others_running(hold.owner):
                return self._hold_check_interval_s  # a running answer is never broken off, not even by a click
        if self._hold.activate(hold, self._clear_all_cards(hold)) is None:
            print(f"[hold] the hold of {hold.owner!r} was released or granted while the cards were cleared for it",
                  file=sys.stderr, flush=True)
        return None

    def _handover_left(self, handover: Handover | None, busy: list[str] | None) -> float | None:
        """How many seconds the ``handover`` still stands; None when there is none or it ended just now: a main
        model is loaded or loading (``busy``, as the hold worker sees it: the person began to load what the cards
        were cleared for, from then on a new hold waits for that model like for any other), or the time is up (the
        person may load nothing at all, and must not keep the cards for ever)."""
        if handover is None:
            return None
        if busy:
            self._end_handover(handover, "model loaded")
            return None
        left = handover.seconds_left()
        if left <= 0:
            self._end_handover(handover, "time is up")
            return None
        return left

    def _end_handover(self, handover: Handover, reason: str) -> None:
        if self._hold.end_handover(handover):
            print(f"[hold] HANDOVER to {handover.for_!r} ended: {reason}", file=sys.stderr, flush=True)

    def _busy_models(self, hold: Hold) -> list[str] | None:
        """The main models that are loaded or loading on any model server (a model announced by a /free counts as
        loading, compared by name only), except the model the hold asks for: the cards are free for ``hold``
        when there are none. Helpers and every other consumer do not count: clearing takes them along. None when
        a model server (or nvidia-smi) cannot answer; that is reported once per change, not at every look."""
        try:
            gpus = self._query()
            busy = [item for server in self._model_servers().values() for item in _main_models(server, hold, gpus)]
            busy += [model for model in self._announced_models() if model != hold.model]
        except (ConsumerError, subprocess.SubprocessError, OSError) as error:
            note = f"cannot tell whether the cards are free: {error}"
            if note != self._cards_note:
                print(f"[hold] {note}", file=sys.stderr, flush=True)
            self._cards_note = note
            return None
        self._cards_note = None
        return busy

    def _clear_all_cards(self, hold: Hold) -> dict[str, Any]:
        """Free every card for ``hold``; the model it asks for keeps running. Besides nvidia-smi, a consumer's
        answer of an unexpected form raises (KeyError, TypeError, AttributeError while reading it); the worker
        must outlive it, or the hold stays requested unseen."""
        keep = () if hold.model is None else (f"{hold.model_server}:{hold.model}",)
        try:
            return self.free(FreeRequest(reason=f"hold: {hold.owner}: {hold.reason}", gpus="all", keep=keep))
        except Exception as error:  # noqa: BLE001 — boundary of the hold worker, the failure goes into the hold
            print(f"[hold] clearing the cards failed:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
            failure = Action("gpu-orc", "all cards", "error", f"clearing failed: {error!r}")
            return {"free": False, "already_running": False, "actions": [_action_json(failure)]}

    def release_hold(self, owner: str, force: bool) -> dict[str, Any]:
        """The owner lets go of the cards (a hold that is only requested is withdrawn). ``force`` lets
        go whoever holds them: for the person, when a program died holding the cards."""
        released = self._hold.release(owner, force, self._handover_grace_s)
        if released is not None and released.owner != owner:
            print(f"[hold] FORCED RELEASE by {owner!r} of the hold of {released.owner!r} "
                  f"({released.state}, {released.reason!r}, since {released.since})", file=sys.stderr, flush=True)
        if released is not None and released.handover_for is not None:
            print(f"[hold] HANDOVER to {released.handover_for!r} after {released.owner!r} let go "
                  f"(until {self._handover_grace_s:g} s)", file=sys.stderr, flush=True)
            self._hold_wake.set()  # the worker must wake at the end of the handover, also when no hold waits
        return {"hold": None}

    def begin_work(self, owner: str, reason: str, ttl_s: float) -> dict[str, Any]:
        """Somebody reports a job on this machine (work.py); reporting again renews the lease."""
        self._work.begin(owner, reason, ttl_s)
        return {"work": self._work.active()}

    def end_work(self, owner: str) -> dict[str, Any]:
        ended = self._work.end(owner)
        self._hold_wake.set()  # a granted hold may be waiting for exactly this
        return {"ended": ended, "work": self._work.active()}

    def work(self) -> dict[str, Any]:
        """The jobs reported right now; answers at once, without nvidia-smi and without asking the consumers."""
        return {"work": self._work.active()}

    def _hold_light(self) -> dict[str, Any]:
        """``held`` only for an ACTIVE hold, so that everybody who reads the light keeps its meaning;
        a requested hold is ``hold_requested`` (whoever starts new work waits on it)."""
        hold = self._hold.get()
        return {
            "held": _hold_json(hold, ACTIVE),
            "hold_requested": _hold_json(hold, REQUESTED),
        }

    def _handover_json(self) -> dict[str, Any] | None:
        handover = self._hold.handover()
        return None if handover is None else handover.as_json()

    def loading(self) -> dict[str, Any]:
        """The traffic light: which big models are loading, and whether the cards are held. While a model
        loads, the cards look emptier than they are; while they are held, somebody keeps them. Either way
        nothing new should start on them. A model counts from the /free that clears for it, before its
        server knows of it. Raises ModelServerUnreachable."""
        models = self._announced_models()
        for server in self._model_servers().values():
            try:
                models.update(server.loading())
            except ConsumerError as error:
                raise ModelServerUnreachable(str(error)) from error
        return {"loading": sorted(models), **self._hold_light()}

    def _announced_models(self) -> set[str]:
        """The models that count as loading from a /free that cleared for them, before their server knows."""
        now = time.monotonic()
        with self._announced_lock:
            return {model for model, until in self._announced.items() if until > now}

    def _announce(self, model: str, until: float) -> None:
        with self._announced_lock:
            self._announced = {m: u for m, u in self._announced.items() if u > time.monotonic()}
            self._announced[model] = until

    def free(self, request: FreeRequest) -> dict[str, Any]:
        started = time.monotonic()
        deadline = started + self._free_timeout_s
        gpus = self._query()
        server = self._model_server(request.model_server) if request.model is not None else None
        targets = self._targets(request, server, gpus)
        self._check_keep(request.keep)
        if server is not None and self._running(server, request.model):
            return self._already_running(gpus, targets, started)
        # A helper loads into its own room: the traffic light stays green for it.
        if server is None or request.model is None or request.dry_run or server.is_helper(request.model):
            return self._free(request, server, targets, gpus, started, deadline)
        self._announce(request.model, deadline)
        try:
            return self._free(request, server, targets, gpus, started, deadline)
        finally:
            self._announce(request.model, time.monotonic() + self._load_grace_s)

    def _free(
        self, request: FreeRequest, server: ModelServer | None, targets: frozenset[str], gpus: Sequence[Gpu],
        started: float, deadline: float,
    ) -> dict[str, Any]:
        with self._gate.enter(deadline) as entered:
            # A request ahead in the queue may have loaded the model meanwhile.
            if entered and server is not None and self._running(server, request.model):
                return self._already_running(gpus, targets, started)
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
            after = report if request.dry_run else self._collect(gpus)
            # A consumer that cannot report may sit on the cards; whether it could, the cards tell: where
            # every MiB belongs to a consumer that did report, there is no room for one that did not.
            cards = gpus if request.dry_run else self._query()
            unknown = _unkept_errors(after, request.keep) if self._unexplained(cards, targets, after) else []
            errors = self._error_actions(report, request.keep) if unknown else []
            actions = self._in_config_order(actions + kept + errors)
            remaining, _ = self._plan(request, server, targets, after.holders)
        failed = any(action.outcome in ("not_freed", "error") for action in actions)
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
        _log(request, result)
        return result

    # --- what to free -------------------------------------------------------------------------

    def _model_servers(self) -> dict[str, ModelServer]:
        return {name: consumer for name, consumer in self._consumers.items() if isinstance(consumer, ModelServer)}

    def _model_server(self, name: str | None) -> ModelServer:
        servers = self._model_servers()
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

    def _unexplained(self, gpus: Sequence[Gpu], targets: frozenset[str], report: _Report) -> bool:
        """Whether a target card holds memory no reporting consumer accounts for (at least
        ``min_freed_mib``, the amount that counts as something), or memory that cannot be accounted for
        because a consumer on it reports none."""
        for gpu in gpus:
            if gpu.uuid not in targets:
                continue
            unattributed = _unattributed_mib(gpu, _on_card(gpu, report.holders))
            if unattributed is None or unattributed >= self._verify.min_freed_mib:
                return True
        return False

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
        return {
            "gpus": [_gpu_json(gpu, report.holders) for gpu in gpus if gpu.uuid in uuids],
            "errors": report.errors,
        }


def _end_process(code: int) -> NoReturn:
    """End the whole service at once from a thread other than the main one (``sys.exit`` would end only it)."""
    os._exit(code)


def _main_models(server: ModelServer, hold: Hold, gpus: Sequence[Gpu]) -> list[str]:
    """The main models of ``server`` that are loaded or loading, except the model ``hold`` asks for.
    Raises ConsumerError."""
    own = (hold.model_server, hold.model) if hold.model is not None else None
    running = [holder.item for holder in server.holders(gpus) if not holder.persistent]
    loading = server.loading()
    return [f"{server.name}:{item}" for item in (*running, *loading) if (server.name, item) != own]


def _hold_json(hold: Hold | None, state: str) -> dict[str, Any] | None:
    return hold.as_json() if hold is not None and hold.state == state else None


def _unkept_errors(report: _Report, keep: Sequence[str]) -> list[dict[str, str]]:
    """The consumers that could not report, except those the request keeps as a whole: what they hold
    stays either way (calibration stops the model server, then has the rest cleared)."""
    return [error for error in report.errors if error["consumer"] not in keep]


def _gpu_json(gpu: Gpu, holders: Mapping[str, Sequence[Holder]]) -> dict[str, Any]:
    """One card of ``/status``."""
    on_card = _on_card(gpu, holders)
    return {
        "index": gpu.index,
        "uuid": gpu.uuid,
        "name": gpu.name,
        "memory_total_mib": gpu.memory_total_mib,
        "memory_used_mib": gpu.memory_used_mib,
        "holders": [
            {
                "consumer": holder.consumer,
                "item": holder.item,
                "memory_mib": holder.memory_mib.get(gpu.uuid),
                "busy": holder.busy,
                "persistent": holder.persistent,
            }
            for holder in on_card
        ],
        "unattributed_mib": _unattributed_mib(gpu, on_card),
    }


def _on_card(gpu: Gpu, holders: Mapping[str, Sequence[Holder]]) -> list[Holder]:
    return [holder for consumer_holders in holders.values() for holder in consumer_holders if gpu.uuid in holder.gpu_uuids]


def _unattributed_mib(gpu: Gpu, on_card: Sequence[Holder]) -> int | None:
    """The card's memory that none of its holders reports; None when a holder on it reports none."""
    reported = [holder.memory_mib.get(gpu.uuid) for holder in on_card]
    if any(mib is None for mib in reported):
        return None
    return max(0, gpu.memory_used_mib - sum(mib for mib in reported if mib is not None))


def _action_json(action: Action) -> dict[str, Any]:
    body: dict[str, Any] = {"consumer": action.consumer, "item": action.item, "outcome": action.outcome}
    if action.detail:
        body["detail"] = action.detail
    if action.waited_s is not None:
        body["waited_s"] = action.waited_s
    return body


def _log(request: FreeRequest, result: Mapping[str, Any]) -> None:
    target = f"model {request.model}" if request.model else f"gpus {request.gpus}"
    outcomes = ", ".join(f"{a['consumer']}:{a['item']}={a['outcome']}" for a in result["actions"]) or "nothing to do"
    mode = " (dry run)" if request.dry_run else ""
    print(f"[free{mode}] {target} — {request.reason} — {outcomes} — free={result['free']} "
          f"in {result['duration_s']} s", flush=True)
