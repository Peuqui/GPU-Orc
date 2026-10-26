"""``gpu-orc serve [--config PATH]`` and ``gpu-orc import-aifred --data DIR [--config PATH]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import measurements
from .config import ConfigError, load_config
from .orchestrator import Orchestrator
from .server import GpuOrcServer

DEFAULT_CONFIG = Path("~/.config/gpu-orc/config.yaml")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gpu-orc", description="Frees GPUs on request.")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="run the HTTP service")
    serve.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=f"default: {DEFAULT_CONFIG}")
    take_over = commands.add_parser(
        "import-aifred", help="take over the measurements of AIfred's calibration (profiles, vision models)")
    take_over.add_argument("--data", type=Path, required=True, help="AIfred's data folder")
    take_over.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=f"default: {DEFAULT_CONFIG}")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config.expanduser())
    except ConfigError as error:
        print(f"gpu-orc: {error}", file=sys.stderr)
        return 2
    store = config.state_dir / measurements.FILE_NAME
    if args.command == "import-aifred":
        return _import_aifred(args.data.expanduser(), store)
    orchestrator = Orchestrator(
        config.consumers, config.free_timeout_s, verify_settings=config.verify,
        measurements=lambda: measurements.load(store), vision_models=config.vision_models,
        place_settings=config.place,
    )
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


def _import_aifred(data_dir: Path, store: Path) -> int:
    """Merge AIfred's measurements into the store: an imported value replaces the one of the same name."""
    try:
        imported = measurements.import_aifred(data_dir)
        current = measurements.load(store)
    except measurements.MeasurementError as error:
        print(f"gpu-orc: {error}", file=sys.stderr)
        return 1
    merged = measurements.Measurements(
        profiles={**current.profiles, **imported.profiles}, vision={**current.vision, **imported.vision})
    measurements.save(store, merged)
    print(f"{len(imported.profiles)} profiles and {len(imported.vision)} vision models taken over into {store}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
