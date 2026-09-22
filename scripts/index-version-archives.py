#!/usr/bin/env python3
"""Write a machine-readable index for version-scoped EdgeForge archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def snapshot(root: Path, path: Path) -> dict[str, object]:
    files = [item for item in path.rglob("*") if item.is_file() and item.name != "ARCHIVE_INDEX.json"]
    checksums = path / "SHA256SUMS"
    gaps = path / "REMOTE_GAPS"
    return {
        "path": str(path.relative_to(root)),
        "file_count": len(files),
        "bytes": sum(item.stat().st_size for item in files),
        "sha256sums": str(checksums.relative_to(root)) if checksums.is_file() else None,
        "sha256sums_digest": digest(checksums) if checksums.is_file() else None,
        "remote_gaps": gaps.read_text(encoding="utf-8").splitlines() if gaps.is_file() else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, default=Path("logs/archive"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.archive_root.resolve()
    output = (args.output or root / "ARCHIVE_INDEX.json").resolve()
    root.mkdir(parents=True, exist_ok=True)
    versions: dict[str, list[dict[str, object]]] = {}
    for version_dir in sorted(root.glob("v*")):
        if not version_dir.is_dir():
            continue
        versions[version_dir.name] = [snapshot(root, child) for child in sorted(version_dir.iterdir()) if child.is_dir()]
    payload = {
        "schema_version": 1,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "archive_root": str(root),
        "versions": versions,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
