#!/usr/bin/env python3
"""Check target/runtime compatibility before attempting model deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from edgeforge.deployment_preflight import evaluate_preflight, load_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--version", default="0.12.0")
    args = parser.parse_args()
    result = evaluate_preflight(load_json(args.manifest), load_json(args.probe))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    log_path = args.log_dir / f"v{args.version}" / "deployment-preflight" / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{result['probe_target']}-{hashlib.sha256(encoded).hexdigest()[:8]}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({"timestamp": time.time(), "version": args.version, "component": "deployment-preflight", "target": result["probe_target"], "result": result}, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
