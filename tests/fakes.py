"""Stand-ins for the consumers' services and for nvidia-smi."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from gpu_orc.gpus import Gpu

Route = Callable[[Any], tuple[int, Any]]

GPUS = [
    Gpu(0, "GPU-aaaa0000", "Card A", 49152, 1000),
    Gpu(1, "GPU-bbbb1111", "Card B", 32768, 500),
    Gpu(2, "GPU-cccc2222", "Card C", 49152, 0),
]


class FakeService:
    """An HTTP server answering ``(method, path with query)`` from ``routes``; records each call."""

    def __init__(self, routes: dict[tuple[str, str], Route]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, str, Any]] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def _handle(self, method: str) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length)) if length else None
                fake.calls.append((method, self.path, body))
                route = fake.routes.get((method, self.path))
                status, answer = (404, {"error": "no route"}) if route is None else route(body)
                # bytes go out as they are: some services answer plain text (llama-swap's unload: "OK").
                data = b"" if answer is None else answer if isinstance(answer, bytes) else json.dumps(answer).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                self._handle("GET")

            def do_POST(self) -> None:
                self._handle("POST")

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"

    @contextmanager
    def running(self) -> Iterator[FakeService]:
        thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        thread.start()
        try:
            yield self
        finally:
            self._server.shutdown()
            self._server.server_close()


def answers(*replies: tuple[int, Any]) -> Route:
    """A route that gives ``replies`` in turn and repeats the last one."""
    remaining = list(replies)

    def route(_body: Any) -> tuple[int, Any]:
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    return route
