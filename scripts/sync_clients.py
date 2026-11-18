"""Copy the GPU-Orc clients byte for byte to the programs listed in clients/consumers.json.

    python scripts/sync_clients.py [--check] [--repos DIR] [--only NAME]

--check changes nothing and exits 1 listing stale or missing copies. --repos is the folder that holds the
repos (default: the parent of this repo), so a sync can go into worktrees first.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = {"python": ROOT / "clients" / "python" / "gpu_orc_client.py", "js": ROOT / "clients" / "js" / "gpu-orc-client.js"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="change nothing; exit 1 when a copy is stale or missing")
    parser.add_argument("--repos", type=Path, default=ROOT.parent, help="folder holding the repos")
    parser.add_argument("--only", help="only this consumer (name in consumers.json)")
    args = parser.parse_args(argv)

    consumers = json.loads((ROOT / "clients" / "consumers.json").read_text(encoding="utf-8"))["consumers"]
    if args.only is not None:
        consumers = [consumer for consumer in consumers if consumer["name"] == args.only]
        if not consumers:
            parser.error(f"no consumer named {args.only}")
    stale = 0
    for consumer in consumers:
        source = SOURCES[consumer["language"]]
        target = args.repos / consumer["repo"] / consumer["target"]
        if not target.exists():
            state = "missing"
        elif target.read_bytes() != source.read_bytes():
            state = "stale"
        else:
            continue
        stale += 1
        if args.check:
            print(f"{state}: {consumer['name']} ({target})")
        else:
            shutil.copyfile(source, target)
            print(f"copied: {consumer['name']} ({target})")
    return 1 if args.check and stale else 0


if __name__ == "__main__":
    sys.exit(main())
