"""``gpu-orc serve [--config PATH]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, load_config
from .orchestrator import Orchestrator
from .server import GpuOrcServer

DEFAULT_CONFIG = Path("~/.config/gpu-orc/config.yaml")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gpu-orc", description="Frees GPUs on request.")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="run the HTTP service")
    serve.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=f"default: {DEFAULT_CONFIG}")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config.expanduser())
    except ConfigError as error:
        print(f"gpu-orc: {error}", file=sys.stderr)
        return 2
    orchestrator = Orchestrator(config.consumers, config.free_timeout_s, verify_settings=config.verify)
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


if __name__ == "__main__":
    sys.exit(main())
