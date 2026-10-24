"""JSON over HTTP to the consumers' services (standard library only)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .model import ConsumerError

REQUEST_TIMEOUT_S = 10.0


def request_json(method: str, url: str, body: Any = None, timeout: float = REQUEST_TIMEOUT_S) -> tuple[int, Any]:
    """Status and decoded JSON body (None when empty). HTTP error statuses are returned, not raised:
    the protocols give some of them a meaning (409 = busy). Raises ConsumerError when the service is
    unreachable or answers something that is not JSON."""
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read()
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise ConsumerError(f"{method} {url}: {error}") from error
    if not raw.strip():
        return status, None
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError as error:
        raise ConsumerError(f"{method} {url}: answer is not JSON (HTTP {status})") from error
