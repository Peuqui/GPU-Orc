"""``gpu-orc serve [--config PATH]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, load_config
from .hold import HoldFileError, HoldStore
from .idle_run import run_if_idle
from .orchestrator import Orchestrator
from .server import GpuOrcServer
from .work import WorkStore

DEFAULT_CONFIG = Path("~/.config/gpu-orc/config.yaml")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gpu-orc", description="Frees GPUs on request.")
    commands = parser.add_subparsers(dest="subcommand", required=True)
    serve = commands.add_parser("serve", help="run the HTTP service")
    serve.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=f"default: {DEFAULT_CONFIG}")
    idle = commands.add_parser(
        "run-if-idle", help="run a command only when no work is reported, e.g. gpu-orc run-if-idle -- /sbin/reboot",
    )
    idle.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=f"for the port, default: {DEFAULT_CONFIG}")
    idle.add_argument("command", nargs=argparse.REMAINDER, help="after --")
    args = parser.parse_args(argv)
    if args.subcommand == "run-if-idle":
        return _run_if_idle(args)
    try:
        config = load_config(args.config.expanduser())
        hold = HoldStore(config.hold_file)
    except (ConfigError, HoldFileError) as error:
        print(f"gpu-orc: {error}", file=sys.stderr)
        return 2
    orchestrator = Orchestrator(
        config.consumers, config.free_timeout_s, verify_settings=config.verify, load_grace_s=config.load_grace_s,
        hold_check_interval_s=config.hold_check_interval_s,
        hold=hold,
        work=WorkStore(),
    )
    orchestrator.start_hold_worker()
    server = GpuOrcServer((config.host, config.port), orchestrator, config.token)
    names = ", ".join(consumer.name for consumer in config.consumers)
    print(f"gpu-orc listening on {config.host}:{config.port}, consumers: {names}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _run_if_idle(args: argparse.Namespace) -> int:
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        print("gpu-orc: run-if-idle needs a command after --", file=sys.stderr)
        return 2
    try:
        config = load_config(args.config.expanduser())
    except ConfigError as error:
        print(f"gpu-orc: {error}", file=sys.stderr)
        return 2
    return run_if_idle(f"http://127.0.0.1:{config.port}", command)


if __name__ == "__main__":
    sys.exit(main())
