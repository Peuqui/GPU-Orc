"""Who holds what, and freeing GPUs: plan from the consumers' own reports, ask them in parallel."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

from .gate import FifoGate
from .gpus import Gpu, GpuSpecError, query_gpus, resolve_gpu
from .model import Action, Consumer, ConsumerError, Holder, ModelServer, UnknownModel
from .verify import DEFAULT_VERIFY, StillHeld, VerifySettings, verify

# How much longer than the overall limit a caller's HTTP timeout should be (answer, nvidia-smi).
CALLER_MARGIN_S = 30.0
# How long a model that was cleared for counts as loading after /free ended, until its server says so.
DEFAULT_LOAD_GRACE_S = 15.0


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
    ) -> None:
        self._consumers = {consumer.name: consumer for consumer in consumers}
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
            **self._status_body(gpus, self._collect(gpus), [gpu.uuid for gpu in gpus]),
        }

    def loading(self) -> dict[str, list[str]]:
        """The traffic light: which big models are loading. While one does, the cards look emptier than
        they are, so nothing else should start on them. A model counts from the /free that clears for
        it, before its server knows of it. Raises ModelServerUnreachable."""
        now = time.monotonic()
        with self._announced_lock:
            models = {model for model, until in self._announced.items() if until > now}
        for server in self._consumers.values():
            if isinstance(server, ModelServer):
                try:
                    models.update(server.loading())
                except ConsumerError as error:
                    raise ModelServerUnreachable(str(error)) from error
        return {"loading": sorted(models)}

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
        _log(request, result)
        return result

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
        return {
            "gpus": [_gpu_json(gpu, report.holders) for gpu in gpus if gpu.uuid in uuids],
            "errors": report.errors,
        }


def _unkept_errors(report: _Report, keep: Sequence[str]) -> list[dict[str, str]]:
    """The consumers that could not report, except those the request keeps as a whole: what they hold
    stays either way (calibration stops the model server, then has the rest cleared)."""
    return [error for error in report.errors if error["consumer"] not in keep]


def _gpu_json(gpu: Gpu, holders: Mapping[str, Sequence[Holder]]) -> dict[str, Any]:
    """One card of ``/status``."""
    on_card = [holder for consumer_holders in holders.values() for holder in consumer_holders if gpu.uuid in holder.gpu_uuids]
    reported = [holder.memory_mib.get(gpu.uuid) for holder in on_card]
    if all(mib is not None for mib in reported):
        unattributed: int | None = max(0, gpu.memory_used_mib - sum(mib for mib in reported if mib is not None))
    else:
        unattributed = None
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
        "unattributed_mib": unattributed,
    }


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
