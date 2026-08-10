"""Immutable content-addressed storage for compiler and kernel artifacts."""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Any


class ArtifactStore:
    def __init__(self, root: str | Path, max_bytes: int = 16 * 1024 * 1024):
        self.root = Path(root).resolve()
        self.max_bytes = max_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, digest: str) -> Path:
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("artifact digest must be a lowercase SHA-256 hex string")
        return self.root / "sha256" / digest[:2] / digest / "blob"

    def put(
        self,
        content: bytes,
        *,
        kind: str,
        media_type: str,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(content, bytes):
            raise TypeError("artifact content must be bytes")
        if len(content) > self.max_bytes:
            raise ValueError(f"artifact exceeds maximum size of {self.max_bytes} bytes")
        if not kind or not media_type or not name:
            raise ValueError("artifact kind, media_type and name are required")
        digest = hashlib.sha256(content).hexdigest()
        path = self._path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.parent / f".upload-{os.getpid()}-{uuid.uuid4().hex}"
            with temporary.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.replace(temporary, path)
            finally:
                if temporary.exists():
                    temporary.unlink()
        return {
            "digest": digest,
            "algorithm": "sha256",
            "size_bytes": len(content),
            "kind": kind,
            "media_type": media_type,
            "name": name,
            "metadata": metadata or {},
            "storage_path": str(path.relative_to(self.root)),
        }

    def read(self, digest: str) -> bytes:
        path = self._path(digest)
        if not path.is_file():
            raise KeyError(digest)
        return path.read_bytes()

