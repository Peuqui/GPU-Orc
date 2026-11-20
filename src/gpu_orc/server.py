"""HTTP interface: GET /health, GET /status, GET /loading, GET /work, POST /free, POST and DELETE /hold, POST /hold/grant, POST /work,
DELETE /work/<owner> (standard library server, one thread per request)."""

from __future__ import annotations

import hmac
import ipaddress
import json
import subprocess
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from . import __version__
from .hold import HoldConflict
from .orchestrator import FreeRequest, ModelServerUnreachable, NotFound, Orchestrator, RequestError

MAX_BODY_BYTES = 64 * 1024
WORK_PATH_PREFIX = "/work/"
HOLD_FIELDS = {"owner", "reason", "model"}
WORK_FIELDS = {"owner", "reason", "ttl_s"}
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
        elif self.path == "/loading":
            self._answer(orchestrator.loading)
        elif self.path == "/work":
            self._answer(orchestrator.work)
        else:
            self._send(HTTPStatus.NOT_FOUND, {"error": f"no such path: {self.path}"})

    def do_POST(self) -> None:
        if not self._authorized():
            return
        if self.path == "/free":
            parse: Any = parse_free_request
            act: Any = self.server.orchestrator.free
        elif self.path == "/hold":
            parse, act = parse_hold_request, lambda request: self.server.orchestrator.hold(*request)
        elif self.path == "/hold/grant":
            parse, act = parse_grant_request, lambda request: self.server.orchestrator.grant_hold(*request)
        elif self.path == "/work":
            parse, act = parse_work_request, lambda request: self.server.orchestrator.begin_work(*request)
        else:
            self._send(HTTPStatus.NOT_FOUND, {"error": f"no such path: {self.path}"})
            return
        try:
            request = parse(self._json_body())
        except RequestError as error:
            self._send(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        self._answer(lambda: act(request))

    def do_DELETE(self) -> None:
        if not self._authorized():
            return
        if self.path == "/hold":
            try:
                owner, force = parse_release_request(self._json_body())
            except RequestError as error:
                self._send(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            self._answer(lambda: self.server.orchestrator.release_hold(owner, force))
        elif self.path.startswith(WORK_PATH_PREFIX) and len(self.path) > len(WORK_PATH_PREFIX):
            owner = urllib.parse.unquote(self.path.removeprefix(WORK_PATH_PREFIX))
            self._answer(lambda: self.server.orchestrator.end_work(owner))
        else:
            self._send(HTTPStatus.NOT_FOUND, {"error": f"no such path: {self.path}"})

    def _answer(self, produce: Any) -> None:
        try:
            self._send(HTTPStatus.OK, produce())
        except RequestError as error:
            self._send(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except HoldConflict as error:
            self._send(HTTPStatus.CONFLICT, {
                "error": str(error), "owner": error.hold.owner, "state": error.hold.state, "since": error.hold.since,
            })
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


def parse_hold_request(body: Any) -> tuple[str, str, str | None, str | None]:
    """``POST /hold``: ``{"owner": "<who holds the cards>", "reason": "<why>", "model": "<model of the model server,
    or null>"}``, plus ``"model_server"`` when several model servers are configured."""
    message = ('body: {"owner": "<who holds the cards>", "reason": "<why>", "model": "<model name or null>", '
               '"model_server": "<name, only with a model>"}')
    if not isinstance(body, dict) or not HOLD_FIELDS <= set(body) <= HOLD_FIELDS | {"model_server"}:
        raise RequestError(message)
    if not all(isinstance(body[key], str) and body[key].strip() for key in ("owner", "reason")):
        raise RequestError("owner and reason must be non-empty strings")
    model, model_server = body["model"], body.get("model_server")
    if model is not None and (not isinstance(model, str) or not model):
        raise RequestError("model must be a name or null")
    if model_server is not None and (model is None or not isinstance(model_server, str) or not model_server):
        raise RequestError("model_server must be a name, and only goes with a model")
    return body["owner"].strip(), body["reason"].strip(), model, model_server


def parse_grant_request(body: Any) -> tuple[str, str]:
    """``POST /hold/grant``: ``{"owner": "<owner of the hold>", "by": "<who grants it>"}``."""
    if not isinstance(body, dict) or set(body) != {"owner", "by"} \
            or not all(isinstance(body[key], str) and body[key].strip() for key in body):
        raise RequestError('body: {"owner": "<owner of the hold>", "by": "<who grants it>"}')
    return body["owner"].strip(), body["by"].strip()


def parse_release_request(body: Any) -> tuple[str, bool]:
    """``DELETE /hold``: ``{"owner": "<who lets go>", "force": true}``; ``force`` (optional) lets go of a hold of another owner."""
    if not isinstance(body, dict) or not {"owner"} <= set(body) <= {"owner", "force"} \
            or not isinstance(body["owner"], str) or not body["owner"].strip() \
            or not isinstance(body.get("force", False), bool):
        raise RequestError('body: {"owner": "<who lets go>", "force": <true to let go of the hold of another owner>}')
    return body["owner"].strip(), body.get("force", False)


def parse_work_request(body: Any) -> tuple[str, str, float]:
    """``POST /work``: ``{"owner": "<who>", "reason": "<what runs>", "ttl_s": <seconds until the lease ends unless renewed>}``."""
    if not isinstance(body, dict) or set(body) != WORK_FIELDS:
        raise RequestError('body: {"owner": "...", "reason": "...", "ttl_s": <seconds>}')
    owner, reason, ttl_s = body["owner"], body["reason"], body["ttl_s"]
    if not all(isinstance(text, str) and text.strip() for text in (owner, reason)):
        raise RequestError("owner and reason must be non-empty strings")
    if isinstance(ttl_s, bool) or not isinstance(ttl_s, (int, float)) or ttl_s <= 0:
        raise RequestError("ttl_s must be a positive number of seconds")
    return owner.strip(), reason.strip(), float(ttl_s)


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
