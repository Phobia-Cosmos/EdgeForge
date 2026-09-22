#!/usr/bin/env python3
"""Collect read-only hardware and runtime evidence for a target device."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from edgeforge.target_probe import probe_target, write_probe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--ssh-host")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=8.0)
    parser.add_argument("--vulkan-icd", help="explicit user-directory Vulkan ICD manifest to record")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--version", default="0.16.3")
    args = parser.parse_args()
    result = probe_target(
        name=args.name,
        ssh_host=args.ssh_host,
        timeout_seconds=args.timeout_seconds,
        vulkan_icd_manifest=args.vulkan_icd,
    )
    write_probe(args.output, result)
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{args.name}-{hashlib.sha256(encoded).hexdigest()[:8]}"
    log_path = args.log_dir / f"v{args.version}" / "target-probe" / f"{run_id}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({"timestamp": time.time(), "version": args.version, "component": "target-probe", "target": args.name, "result_digest": hashlib.sha256(encoded).hexdigest(), "result": result}, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
