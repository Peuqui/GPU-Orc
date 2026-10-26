"""llama-swap: a model server that starts one process per model.

``/running`` lists the loaded models; their GPUs come from ``CUDA_VISIBLE_DEVICES`` in the model's
``env`` in llama-swap's own config. Members of groups with ``persistent: true`` stay loaded when
another model starts, so they are the ones GPU-Orc must take down before a load. Other models
llama-swap swaps out by itself. A vLLM model may fill ``--gpu-memory-utilization`` of each of its
cards (its ``cmd``) whether that memory is allocated yet or not.
"""

from __future__ import annotations

import fnmatch
import time
import urllib.parse
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..gpus import Gpu, visible_gpu_uuids
from ..http_json import request_json, request_status
from ..measurements import Measurements
from ..model import Action, ConsumerError, Holder, UnknownModel
from ..profiles import Profile, memory_fraction, read_profiles


@dataclass(frozen=True)
class _SwapConfig:
    """llama-swap's config as GPU-Orc reads it: the raw entries, the environment of each, the members of
    persistent groups and the share of its cards each vLLM entry may fill."""

    entries: dict[str, dict[str, Any]]
    envs: dict[str, dict[str, str]]
    persistent: frozenset[str]
    fractions: dict[str, float]


@dataclass
class LlamaSwap:
    name: str
    url: str
    config_file: Path
    no_eviction_models: Sequence[str] = field(default_factory=list)

    def holders(self, gpus: Sequence[Gpu]) -> list[Holder]:
        config = self._read_config()
        holders = []
        for model in self._running():
            if model not in config.envs:
                raise ConsumerError(f"{self.name}: running model {model!r} is not in {self.config_file}")
            try:
                cards = visible_gpu_uuids(config.envs[model], gpus)
            except ValueError as error:
                raise ConsumerError(f"{self.name}: {model}: {error}") from error
            holders.append(Holder(
                consumer=self.name, item=model, gpu_uuids=cards, persistent=model in config.persistent,
                memory_fraction=config.fractions.get(model),
            ))
        return holders

    def profiles(self, gpus: Sequence[Gpu], measurements: Measurements) -> list[Profile]:
        """Every entry of the config as a profile (see :mod:`gpu_orc.profiles`)."""
        config = self._read_config()
        return read_profiles(config.entries, config.envs, config.persistent, gpus, measurements)

    def release(self, holders: Sequence[Holder], *, reason: str, deadline: float) -> list[Action]:
        actions = []
        for holder in holders:
            started = time.monotonic()
            model = urllib.parse.quote(holder.item, safe="")
            timeout = max(1.0, deadline - started)
            status = request_status("POST", f"{self.url}/api/models/unload/{model}", timeout=timeout)
            waited = round(time.monotonic() - started, 1)
            if status != 200:
                actions.append(Action(self.name, holder.item, "error", f"unload answered HTTP {status}", waited))
            elif holder.item in self._running():
                actions.append(Action(self.name, holder.item, "error", "still running after unload", waited))
            else:
                actions.append(Action(self.name, holder.item, "released", "", waited))
        return actions

    def is_running(self, model: str) -> bool:
        return model in self._running()

    def model_gpu_uuids(self, model: str, gpus: Sequence[Gpu]) -> frozenset[str]:
        envs = self._read_config().envs
        if model not in envs:
            raise UnknownModel(model)
        return visible_gpu_uuids(envs[model], gpus)

    def evicts_persistent(self, model: str) -> bool:
        if model in self._read_config().persistent:
            return False
        return not any(fnmatch.fnmatchcase(model, pattern) for pattern in self.no_eviction_models)

    def _running(self) -> list[str]:
        status, body = request_json("GET", f"{self.url}/running")
        if status != 200 or not isinstance(body, dict):
            raise ConsumerError(f"{self.name}: /running answered HTTP {status}")
        return [entry["model"] for entry in body.get("running") or [] if entry.get("model")]

    def _read_config(self) -> _SwapConfig:
        """Read on every call: the config changes while the server runs (calibration, new models)."""
        try:
            config: Any = yaml.safe_load(self.config_file.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ConsumerError(f"{self.name}: cannot read {self.config_file}: {error}") from error
        entries = dict(config.get("models") or {})
        fractions = {
            name: fraction for name, entry in entries.items()
            if (fraction := memory_fraction(str(entry.get("cmd") or ""))) is not None
        }
        persistent = frozenset(
            member
            for group in (config.get("groups") or {}).values()
            if group.get("persistent")
            for member in group.get("members") or []
        )
        return _SwapConfig(entries, {name: _env(entry) for name, entry in entries.items()}, persistent, fractions)


def _env(entry: dict[str, Any]) -> dict[str, str]:
    pairs = (item.split("=", 1) for item in entry.get("env") or [] if isinstance(item, str) and "=" in item)
    return {key.strip(): value.strip() for key, value in pairs}

