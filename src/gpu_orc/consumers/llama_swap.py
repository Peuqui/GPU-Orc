"""llama-swap: a model server that starts one process per model.

``/running`` lists the loaded models; their GPUs come from ``CUDA_VISIBLE_DEVICES`` in the model's
``env`` in llama-swap's own config. Members of groups with ``persistent: true`` stay loaded when
another model starts, so they are the ones GPU-Orc must take down before a load. Other models
llama-swap swaps out by itself.
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
from ..model import Action, ConsumerError, Holder, UnknownModel


@dataclass
class LlamaSwap:
    name: str
    url: str
    config_file: Path
    no_eviction_models: Sequence[str] = field(default_factory=list)

    def holders(self, gpus: Sequence[Gpu]) -> list[Holder]:
        models, persistent = self._read_config()
        holders = []
        for model in self._running():
            if model not in models:
                raise ConsumerError(f"{self.name}: running model {model!r} is not in {self.config_file}")
            try:
                cards = visible_gpu_uuids(models[model], gpus)
            except ValueError as error:
                raise ConsumerError(f"{self.name}: {model}: {error}") from error
            holders.append(Holder(consumer=self.name, item=model, gpu_uuids=cards, persistent=model in persistent))
        return holders

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
        models, _ = self._read_config()
        if model not in models:
            raise UnknownModel(model)
        return visible_gpu_uuids(models[model], gpus)

    def evicts_persistent(self, model: str) -> bool:
        _, persistent = self._read_config()
        if model in persistent:
            return False
        return not any(fnmatch.fnmatchcase(model, pattern) for pattern in self.no_eviction_models)

    def _running(self) -> list[str]:
        status, body = request_json("GET", f"{self.url}/running")
        if status != 200 or not isinstance(body, dict):
            raise ConsumerError(f"{self.name}: /running answered HTTP {status}")
        return [entry["model"] for entry in body.get("running") or [] if entry.get("model")]

    def _read_config(self) -> tuple[dict[str, dict[str, str]], frozenset[str]]:
        """Environment per model and the members of persistent groups. Read on every call: the
        config changes while the server runs (calibration, new models)."""
        try:
            config: Any = yaml.safe_load(self.config_file.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ConsumerError(f"{self.name}: cannot read {self.config_file}: {error}") from error
        models = {name: _env(entry) for name, entry in (config.get("models") or {}).items()}
        persistent = frozenset(
            member
            for group in (config.get("groups") or {}).values()
            if group.get("persistent")
            for member in group.get("members") or []
        )
        return models, persistent


def _env(entry: dict[str, Any]) -> dict[str, str]:
    pairs = (item.split("=", 1) for item in entry.get("env") or [] if isinstance(item, str) and "=" in item)
    return {key.strip(): value.strip() for key, value in pairs}
