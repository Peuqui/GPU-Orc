"""``gpu-orc run-if-idle -- <command>``: run the command only when no work is reported (work.py).

For the weekly reboot: ``/etc/cron.d`` calls this instead of ``/sbin/reboot``. Work running means the command is
skipped on purpose (exit 0, the next date tries again, there is no waiting and no deadline). GPU-Orc not
answering means the machine cannot say whether work runs: the command is skipped as well, but that is an
error (exit 3, on stderr), not a decision.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Sequence
from typing import Any

from .http_json import request_json
from .model import ConsumerError

EXIT_GPU_ORC_UNKNOWN = 3


def run_if_idle(
    base_url: str,
    command: Sequence[str],
    run: Callable[[Sequence[str]], Any] = subprocess.call,
) -> int:
    """The command's exit code when it ran; 0 when work is running (command skipped); 3 when GPU-Orc says nothing."""
    try:
        status, body = request_json("GET", f"{base_url}/work")
        work = body["work"]
    except (ConsumerError, KeyError, TypeError) as error:
        print(f"gpu-orc: cannot tell whether work runs, {command[0]} skipped: {error!r}", file=sys.stderr)
        return EXIT_GPU_ORC_UNKNOWN
    if status != 200:
        print(f"gpu-orc: GET /work answered HTTP {status}, {command[0]} skipped", file=sys.stderr)
        return EXIT_GPU_ORC_UNKNOWN
    if work:
        owners = ", ".join(f"{entry['owner']} ({entry['reason']}, {entry['running_for_s']:.0f} s)" for entry in work)
        print(f"gpu-orc: work is running, {command[0]} skipped: {owners}", flush=True)
        return 0
    return int(run(command))
