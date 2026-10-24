"""HTTP interface: GET /health, GET /status, POST /free (standard library server, one thread per request)."""

from __future__ import annotations

import hmac
import ipaddress
import json
import subprocess
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from . import __version__
from .orchestrator import FreeRequest, ModelServerUnreachable, NotFound, Orchestrator, RequestError

MAX_BODY_BYTES = 64 * 1024
FREE_FIELDS = {"gpus", "model", "model_server", "keep", "reason", "dry_run"}


class GpuOrcServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], orchestrator: Orchestrator, token: str | None) -> None:
        super().__init__(address, Handler)
        self.orchestrator = orchestrator
        self.token = token


class Handler(BaseHTTPRequestHandler):
    server: GpuOrcServer

    def do_GET(self) -> None:
        if not self._authorized():
            return
        orchestrator = self.server.orchestrator
        if self.path == "/health":
            self._send(HTTPStatus.OK, {
                "status": "ok",
                "version": __version__,
                "free_timeout_s": orchestrator.free_timeout_s,
                "caller_timeout_s": orchestrator.caller_timeout_s,
            })
        elif self.path == "/status":
            self._answer(orchestrator.status)
        else:
            self._send(HTTPStatus.NOT_FOUND, {"error": f"no such path: {self.path}"})

    def do_POST(self) -> None:
        if not self._authorized():
            return
        if self.path != "/free":
            self._send(HTTPStatus.NOT_FOUND, {"error": f"no such path: {self.path}"})
            return
        try:
            request = parse_free_request(self._json_body())
        except RequestError as error:
            self._send(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        self._answer(lambda: self.server.orchestrator.free(request))

    def _answer(self, produce: Any) -> None:
        try:
            self._send(HTTPStatus.OK, produce())
        except RequestError as error:
            self._send(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except NotFound as error:
            self._send(HTTPStatus.NOT_FOUND, {"error": str(error)})
        except ModelServerUnreachable as error:
            self._send(HTTPStatus.BAD_GATEWAY, {"error": f"model server unreachable: {error}"})
        except (subprocess.SubprocessError, OSError) as error:
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"nvidia-smi failed: {error}"})

    def _authorized(self) -> bool:
        refusal = authorization_error(self.client_address[0], self.headers.get("Authorization", ""), self.server.token)
        if refusal is None:
            return True
        status, message = refusal
        self._send(status, {"error": message})
        return False

    def _json_body(self) -> Any:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            raise RequestError("request body too large")
        try:
            return json.loads(self.rfile.read(length) or b"null")
        except json.JSONDecodeError as error:
            raise RequestError(f"body is not JSON: {error}") from error

    def _send(self, status: HTTPStatus, body: Any) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        # The hook asks on every prompt; /free logs its own line, the rest stays quiet.
        pass


def authorization_error(client_ip: str, header: str, token: str | None) -> tuple[HTTPStatus, str] | None:
    """None when the request may pass: from loopback always, from elsewhere only with the token."""
    if ipaddress.ip_address(client_ip).is_loopback:
        return None
    if not header.startswith("Bearer "):
        return HTTPStatus.UNAUTHORIZED, "token required"
    if token is None or not hmac.compare_digest(header.removeprefix("Bearer ").encode(), token.encode()):
        return HTTPStatus.FORBIDDEN, "wrong token"
    return None


def parse_free_request(body: Any) -> FreeRequest:
    if not isinstance(body, dict):
        raise RequestError("body must be a JSON object")
    unknown = set(body) - FREE_FIELDS
    if unknown:
        raise RequestError(f"unknown fields: {sorted(unknown)}")
    has_gpus, has_model = "gpus" in body, "model" in body
    if has_gpus == has_model:
        raise RequestError("give exactly one of gpus and model")
    reason = body.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise RequestError("reason is required")
    keep = body.get("keep", [])
    if not isinstance(keep, list) or not all(isinstance(entry, str) and entry for entry in keep):
        raise RequestError("keep must be a list of names")
    dry_run = body.get("dry_run", False)
    if not isinstance(dry_run, bool):
        raise RequestError("dry_run must be true or false")
    if has_model:
        model, model_server = body["model"], body.get("model_server")
        if not isinstance(model, str) or not model:
            raise RequestError("model must be a name")
        if model_server is not None and (not isinstance(model_server, str) or not model_server):
            raise RequestError("model_server must be a name")
        return FreeRequest(reason=reason, model=model, model_server=model_server, keep=tuple(keep), dry_run=dry_run)
    if "model_server" in body:
        raise RequestError("model_server belongs to the model form")
    gpus = body["gpus"]
    if gpus == "all":
        return FreeRequest(reason=reason, gpus="all", keep=tuple(keep), dry_run=dry_run)
    if (
        not isinstance(gpus, list)
        or not gpus
        or not all(isinstance(spec, int | str) and not isinstance(spec, bool) for spec in gpus)
    ):
        raise RequestError('gpus must be "all" or a non-empty list of indices or UUIDs')
    return FreeRequest(reason=reason, gpus=tuple(gpus), keep=tuple(keep), dry_run=dry_run)
