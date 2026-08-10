"""Small JSON HTTP client shared by the CLI and worker."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class APIError(RuntimeError):
    pass


class Client:
    def __init__(self, base_url: str, token: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def request(self, method: str, path: str, data: dict[str, Any] | None = None) -> Any:
        body = None
        headers = {"Accept": "application/json", "Authorization": f"Bearer {self.token}"}
        if data is not None:
            body = json.dumps(data, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(self.base_url + path, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except HTTPError as error:
            payload = error.read().decode("utf-8", errors="replace")
            try:
                message = json.loads(payload).get("error", payload)
            except json.JSONDecodeError:
                message = payload
            raise APIError(f"HTTP {error.code}: {message}") from error
        except URLError as error:
            raise APIError(f"control plane unavailable: {error.reason}") from error

