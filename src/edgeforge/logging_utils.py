"""Version-scoped structured logging with one immutable file per process."""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any


class JsonlFileHandler(logging.Handler):
    def __init__(self, path: Path, component: str, version: str, run_id: str):
        super().__init__()
        self.path = path
        self.component = component
        self.version = version
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8", buffering=1)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            item: dict[str, Any] = {
                "timestamp": time.time(),
                "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "component": self.component,
                "version": self.version,
                "run_id": self.run_id,
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
            }
            if record.exc_info:
                item["exception"] = "".join(traceback.format_exception(*record.exc_info))
            self._stream.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        try:
            self._stream.close()
        finally:
            super().close()


def configure_logging(component: str, version: str, log_dir: str | Path, level: str = "INFO") -> Path:
    """Configure console plus an append-only, version-scoped JSONL file."""
    run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    directory = Path(log_dir) / f"v{version}" / component
    path = directory / f"{run_id}.jsonl"
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(stream)
    root.addHandler(JsonlFileHandler(path, component, version, run_id))
    logging.getLogger("edgeforge").info("versioned log file: %s", path)
    return path

