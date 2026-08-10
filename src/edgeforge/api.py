"""HTTP control plane API built only on the Python standard library."""

from __future__ import annotations

import hmac
import json
import logging
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from edgeforge import __version__
from edgeforge.db import Store


LOG = logging.getLogger("edgeforge.control")
TASK_PATH = re.compile(r"^/api/v1/tasks/([0-9a-f]+)$")
COMPLETE_PATH = re.compile(r"^/api/v1/tasks/([0-9a-f]+)/complete$")
HEARTBEAT_PATH = re.compile(r"^/api/v1/workers/([^/]+)/heartbeat$")
LEASE_PATH = re.compile(r"^/api/v1/workers/([^/]+)/lease$")


class ControlServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], store: Store, token: str):
        super().__init__(address, ControlHandler)
        self.store = store
        self.token = token


class ControlHandler(BaseHTTPRequestHandler):
    server: ControlServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        message = format % args
        if ("/heartbeat " in message or "/lease " in message) and " 200 " in message:
            LOG.debug("%s - %s", self.address_string(), message)
        else:
            LOG.info("%s - %s", self.address_string(), message)

    def _send(self, status: int, data: Any) -> None:
        body = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        expected = f"Bearer {self.server.token}"
        supplied = self.headers.get("Authorization", "")
        return bool(self.server.token) and hmac.compare_digest(supplied, expected)

    def _require_auth(self) -> bool:
        if self._authorized():
            return True
        self._send(HTTPStatus.UNAUTHORIZED, {"error": "missing or invalid bearer token"})
        return False

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("invalid Content-Length") from error
        if length <= 0 or length > 1_048_576:
            raise ValueError("request body must be between 1 byte and 1 MiB")
        try:
            value = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as error:
            raise ValueError("request body is not valid JSON") from error
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/healthz":
            self._send(HTTPStatus.OK, {"status": "ok", "version": __version__})
            return
        if not self._require_auth():
            return
        try:
            if parsed.path == "/api/v1/workers":
                self._send(HTTPStatus.OK, {"workers": self.server.store.list_workers()})
                return
            if parsed.path == "/api/v1/tasks":
                query = parse_qs(parsed.query)
                limit = int(query.get("limit", ["100"])[0])
                self._send(HTTPStatus.OK, {"tasks": self.server.store.list_tasks(limit)})
                return
            if parsed.path == "/api/v1/events":
                query = parse_qs(parsed.query)
                limit = int(query.get("limit", ["100"])[0])
                version = query.get("version", [None])[0]
                self._send(HTTPStatus.OK, {"events": self.server.store.list_events(version, limit)})
                return
            if parsed.path == "/api/v1/benchmarks":
                query = parse_qs(parsed.query)
                limit = int(query.get("limit", ["100"])[0])
                operator = query.get("operator", [None])[0]
                self._send(HTTPStatus.OK, {"benchmarks": self.server.store.list_benchmarks(operator, limit)})
                return
            if parsed.path == "/api/v1/releases":
                self._send(HTTPStatus.OK, {"releases": self.server.store.list_releases()})
                return
            match = TASK_PATH.match(parsed.path)
            if match:
                self._send(HTTPStatus.OK, self.server.store.get_task(match.group(1)))
                return
            self._send(HTTPStatus.NOT_FOUND, {"error": "route not found"})
        except KeyError:
            self._send(HTTPStatus.NOT_FOUND, {"error": "resource not found"})
        except (TypeError, ValueError) as error:
            self._send(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except Exception:
            LOG.exception("unhandled GET error")
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal server error"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if not self._require_auth():
            return
        try:
            data = self._read_json()
            if parsed.path == "/api/v1/workers/register":
                if not data.get("id"):
                    raise ValueError("worker id is required")
                worker = self.server.store.register_worker(data)
                self._send(HTTPStatus.OK, {"worker": worker, "heartbeat_interval": 5})
                return
            if parsed.path == "/api/v1/tasks":
                task = self.server.store.create_task(data)
                self._send(HTTPStatus.CREATED, task)
                return
            if parsed.path == "/api/v1/releases":
                version = str(data.get("version", ""))
                summary = str(data.get("summary", ""))
                if not version or not summary:
                    raise ValueError("release version and summary are required")
                release = self.server.store.record_release(version, summary, data.get("metadata") or {}, data.get("status", "active"))
                self._send(HTTPStatus.CREATED, release)
                return
            match = HEARTBEAT_PATH.match(parsed.path)
            if match:
                worker = self.server.store.heartbeat(match.group(1), data.get("metrics") or {})
                self._send(HTTPStatus.OK, {"worker": worker})
                return
            match = LEASE_PATH.match(parsed.path)
            if match:
                task = self.server.store.lease_task(match.group(1))
                self._send(HTTPStatus.OK, {"task": task})
                return
            match = COMPLETE_PATH.match(parsed.path)
            if match:
                worker_id = data.get("worker_id")
                if not worker_id:
                    raise ValueError("worker_id is required")
                task = self.server.store.complete_task(match.group(1), worker_id, data)
                self._send(HTTPStatus.OK, task)
                return
            self._send(HTTPStatus.NOT_FOUND, {"error": "route not found"})
        except KeyError:
            self._send(HTTPStatus.NOT_FOUND, {"error": "resource not found"})
        except (RuntimeError, TypeError, ValueError) as error:
            self._send(HTTPStatus.CONFLICT if isinstance(error, RuntimeError) else HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except Exception:
            LOG.exception("unhandled POST error")
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal server error"})


def serve(bind: str, port: int, database: str, token: str, worker_timeout: float) -> None:
    store = Store(database, worker_timeout=worker_timeout)
    server = ControlServer((bind, port), store, token)
    LOG.info("control plane listening on http://%s:%d", bind, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
