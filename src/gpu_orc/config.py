"""The service's YAML config: where it listens, its token, its time limit and the consumers."""

from __future__ import annotations

import ipaddress
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .consumers.gpu_release import GpuRelease
from .consumers.llama_swap import LlamaSwap
from .model import Consumer
from .verify import DEFAULT_VERIFY, VerifySettings

DEFAULT_PORT = 5070
DEFAULT_FREE_TIMEOUT_S = 180.0


class ConfigError(ValueError):
    """The config file is missing a value or has one of the wrong kind."""


@dataclass(frozen=True)
class Config:
    host: str
    port: int
    token: str | None
    free_timeout_s: float
    verify: VerifySettings
    consumers: tuple[Consumer, ...]


def load_config(path: Path) -> Config:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"cannot read {path}: {error}") from error
    data = _mapping(raw, str(path))
    listen = _mapping(data.get("listen", {}), "listen")
    host = _get(listen, "host", str, "listen", "127.0.0.1")
    port = _get(listen, "port", int, "listen", DEFAULT_PORT)
    token = _read_token(data.get("token_file"))
    if token is None and not _is_loopback(host):
        raise ConfigError(f"listen.host {host!r} is not loopback: token_file is required")
    free_timeout_s = float(_get(data, "free_timeout_s", (int, float), "config", DEFAULT_FREE_TIMEOUT_S))
    defaults = DEFAULT_VERIFY
    verify = VerifySettings(
        timeout_s=float(_get(data, "verify_timeout_s", (int, float), "config", defaults.timeout_s)),
        min_freed_mib=_get(data, "min_freed_mib", int, "config", defaults.min_freed_mib),
    )
    entries = data.get("consumers")
    if not isinstance(entries, list) or not entries:
        raise ConfigError("consumers: a non-empty list is required")
    consumers = tuple(_consumer(_mapping(entry, "consumers[]")) for entry in entries)
    names = [consumer.name for consumer in consumers]
    if len(set(names)) != len(names):
        raise ConfigError(f"consumers: names must be unique, got {names}")
    return Config(host, port, token, free_timeout_s, verify, consumers)


def _consumer(entry: Mapping[str, Any]) -> Consumer:
    name = _get(entry, "name", str, "consumer")
    kind = _get(entry, "type", str, name)
    builder = _BUILDERS.get(kind)
    if builder is None:
        raise ConfigError(f"{name}: unknown type {kind!r}, known: {sorted(_BUILDERS)}")
    known = {"name", "type", *_FIELDS[kind]}
    unknown = set(entry) - known
    if unknown:
        raise ConfigError(f"{name}: unknown keys {sorted(unknown)} for type {kind!r}")
    return builder(name, entry)


def _gpu_release(name: str, entry: Mapping[str, Any]) -> Consumer:
    return GpuRelease(
        name=name,
        url=_url(entry, name),
        busy_wait_s=float(_get(entry, "busy_wait_s", (int, float), name, 0)),
        force_after_wait=_get(entry, "force_after_wait", bool, name, False),
    )


def _llama_swap(name: str, entry: Mapping[str, Any]) -> Consumer:
    patterns = _get(entry, "no_eviction_models", list, name, [])
    if not all(isinstance(pattern, str) for pattern in patterns):
        raise ConfigError(f"{name}: no_eviction_models must be a list of strings")
    return LlamaSwap(
        name=name,
        url=_url(entry, name),
        config_file=Path(_get(entry, "config_file", str, name)).expanduser(),
        no_eviction_models=tuple(patterns),
    )


_BUILDERS: dict[str, Callable[[str, Mapping[str, Any]], Consumer]] = {
    "gpu-release": _gpu_release,
    "llama-swap": _llama_swap,
}
_FIELDS: dict[str, tuple[str, ...]] = {
    "gpu-release": ("url", "busy_wait_s", "force_after_wait"),
    "llama-swap": ("url", "config_file", "no_eviction_models"),
}

_MISSING = object()


def _get(data: Mapping[str, Any], key: str, kind: type | tuple[type, ...], where: str, default: Any = _MISSING) -> Any:
    if key not in data:
        if default is _MISSING:
            raise ConfigError(f"{where}: {key} is required")
        return default
    value = data[key]
    # bool is an int: a port or a time must not be true/false.
    if isinstance(value, bool) and kind is not bool:
        raise ConfigError(f"{where}: {key} has the wrong type")
    if not isinstance(value, kind):
        raise ConfigError(f"{where}: {key} has the wrong type")
    return value


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: a mapping is required")
    return value


def _url(entry: Mapping[str, Any], name: str) -> str:
    return str(_get(entry, "url", str, name)).rstrip("/")


def _read_token(token_file: Any) -> str | None:
    if token_file is None:
        return None
    if not isinstance(token_file, str):
        raise ConfigError("token_file must be a path")
    path = Path(token_file).expanduser()
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ConfigError(f"cannot read token_file {path}: {error}") from error
    if not token:
        raise ConfigError(f"token_file {path} is empty")
    return token


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
