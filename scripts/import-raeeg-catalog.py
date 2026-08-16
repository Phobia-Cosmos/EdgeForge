#!/usr/bin/env python3
"""Submit the versioned local RA-EEG result catalog to EdgeForge."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

from edgeforge.client import Client


def merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="config/raeeg-local-catalog.json")
    parser.add_argument("--control-url", default=os.environ.get("EDGEFORGE_CONTROL_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--token", default=os.environ.get("EDGEFORGE_TOKEN"))
    parser.add_argument("--worker-id", default="worker-4070s")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    if not args.token:
        parser.error("--token or EDGEFORGE_TOKEN is required")
    catalog = json.loads(Path(args.catalog).read_text(encoding="utf-8"))
    if catalog.get("schema_version") != 1 or not isinstance(catalog.get("experiments"), list):
        parser.error("unsupported catalog schema")
    client = Client(args.control_url, args.token)
    submitted = []
    for entry in catalog["experiments"]:
        spec = merge(catalog.get("defaults") or {}, entry)
        task = client.request(
            "POST",
            "/api/v1/tasks",
            {
                "kind": "experiment_run",
                "payload": {"spec": spec},
                "requirements": {"worker_ids": [args.worker_id]},
                "priority": 10,
            },
        )
        submitted.append({"experiment_id": spec["experiment_id"], "task_id": task["id"]})
    if args.wait:
        for item in submitted:
            while True:
                task = client.request("GET", f"/api/v1/tasks/{item['task_id']}")
                if task["status"] in {"succeeded", "failed", "cancelled"}:
                    item["status"] = task["status"]
                    item["error"] = task.get("error")
                    break
                time.sleep(0.5)
    print(json.dumps({"submitted": submitted}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
