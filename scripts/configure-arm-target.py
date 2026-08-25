#!/usr/bin/env python3
"""Prepare user-owned EdgeForge worker directories on ARM/RISC-V targets.

The default mode is read-only inventory.  ``--apply`` creates only user-owned
directories and templates; it never installs packages, writes credentials, or
advertises an accelerator that has not passed a target probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


VERSION = "0.16.3"
REMOTE_SOURCE = "~/.local/src/edgeforge"
REMOTE_BASE = "~/.local/share/edgeforge"

TARGETS: dict[str, dict[str, str]] = {
    "orangepi": {"ssh_host": "orangepi", "architecture": "aarch64", "role": "arm64-cpu-reference"},
    "p550": {"ssh_host": "p550", "architecture": "riscv64", "role": "riscv64-reference-build"},
}


REMOTE_INVENTORY = r'''
import glob
import importlib.util
import json
import os
import platform
import shutil
import subprocess


def command(name):
    path = shutil.which(name)
    if not path:
        return None
    try:
        result = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=5, check=False)
        return {"path": path, "version": (result.stdout or result.stderr).splitlines()[0] if (result.stdout or result.stderr) else None}
    except Exception as error:
        return {"path": path, "version": None, "error": str(error)}


def present(path):
    return os.path.exists(path)


result = {
    "hostname": platform.node(),
    "architecture": platform.machine(),
    "kernel": platform.release(),
    "python": platform.python_version(),
    "user": os.environ.get("USER"),
    "home": os.path.expanduser("~"),
    "commands": {name: command(name) for name in ("python3", "pip3", "git", "gcc", "clang", "cmake", "ninja", "systemctl")},
    "python_modules": {name: bool(importlib.util.find_spec(name)) for name in ("numpy", "torch", "onnxruntime", "iree", "rknn")},
    "rknn_runtime": {
        "server": command("rknn_server"),
        "demo": command("rknn_demo"),
        "libraries": [path for path in ("/usr/lib/librknnrt.so", "/usr/lib/librknn_api.so") if present(path)],
        "models": sorted(glob.glob("/usr/share/rknn_demo/*.rknn")),
        "device_nodes": sorted(glob.glob("/dev/rknpu*")),
    },
    "drm_nodes": sorted(glob.glob("/dev/dri/card*") + glob.glob("/dev/dri/renderD*")),
    "drm_driver_map": {
        path: os.path.basename(os.path.realpath(path))
        for path in sorted(glob.glob("/sys/class/drm/card*/device/driver") + glob.glob("/sys/class/drm/renderD*/device/driver"))
        if os.path.exists(path)
    },
    "npu_platform_devices": sorted(glob.glob("/sys/bus/platform/devices/*npu*")),
    "gpu_userspace": {
        "opencl_icd": sorted(glob.glob("/etc/OpenCL/vendors/*.icd")),
        "vulkan_icd": sorted(glob.glob("/etc/vulkan/icd.d/*.json") + glob.glob("/usr/share/vulkan/icd.d/*.json")),
        "libraries": [path for path in ("/usr/lib/aarch64-linux-gnu/libmali.so.1", "/usr/lib/aarch64-linux-gnu/libOpenCL.so.1", "/usr/lib/aarch64-linux-gnu/libvulkan.so.1") if present(path)],
    },
}
print(json.dumps(result, sort_keys=True))
'''


def _ssh(host: str, script: str, timeout: float) -> dict[str, Any]:
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, shlex.join(["python3", "-c", script])]
    started = time.perf_counter()
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "unreachable", "exit_code": None, "stdout": "", "stderr": str(error), "elapsed_ms": round((time.perf_counter() - started) * 1000, 3)}
    return {
        "status": "succeeded" if completed.returncode == 0 else "failed",
        "exit_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def _rsync(source_root: Path, host: str, destination: str, timeout: float) -> dict[str, Any]:
    excludes = [
        ".git/",
        ".edgeforge/",
        "logs/",
        "data/",
        "artifacts/",
        "__pycache__/",
        ".venv/",
        "*.pyc",
    ]
    argv = ["rsync", "-a"]
    for item in excludes:
        argv.extend(["--exclude", item])
    argv.extend([f"{source_root.resolve()}/", f"{host}:{destination}/"])
    started = time.perf_counter()
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "unreachable", "exit_code": None, "stdout": "", "stderr": str(error), "elapsed_ms": round((time.perf_counter() - started) * 1000, 3)}
    return {
        "status": "succeeded" if completed.returncode == 0 else "failed",
        "exit_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def _remote_prepare(target: str, profile: dict[str, str]) -> str:
    payload = json.dumps({"target": target, **profile}, sort_keys=True)
    return f'''
from pathlib import Path
import hashlib
import json
import os
import stat

profile = json.loads({payload!r})
home = Path.home()
source = home / ".local" / "src" / "edgeforge"
base = home / ".local" / "share" / "edgeforge"
state = home / ".local" / "state" / "edgeforge"
config = home / ".config" / "edgeforge"
for path in (source, base / "work", state / "logs", config):
    path.mkdir(parents=True, exist_ok=True)

worker_id = "edgeforge-" + profile["target"]
env = "\\n".join([
    "# Template only: put the real token in a protected environment, not in this file.",
    "EDGEFORGE_CONTROL_URL=http://127.0.0.1:8080",
    "EDGEFORGE_TOKEN=replace-with-runtime-secret",
    f"EDGEFORGE_WORKER_ID={{worker_id}}",
    "EDGEFORGE_BACKENDS=python-reference",
    f"EDGEFORGE_WORK_ROOT={{base / 'work'}}",
    f"EDGEFORGE_LOG_DIR={{state / 'logs'}}",
    "EDGEFORGE_LOG_LEVEL=INFO",
]) + "\\n"
(config / "edgeforge.env.example").write_text(env, encoding="utf-8")
(config / "edgeforge.env.example").chmod(stat.S_IRUSR | stat.S_IWUSR)
runner = "\\n".join([
    "#!/bin/sh",
    "set -eu",
    f"export PYTHONPATH={{source / 'src'}}${{{{PYTHONPATH:+:$PYTHONPATH}}}}",
    f"export EDGEFORGE_WORK_ROOT={{base / 'work'}}",
    f"export EDGEFORGE_LOG_DIR={{state / 'logs'}}",
    "export EDGEFORGE_BACKENDS=${{EDGEFORGE_BACKENDS:-python-reference}}",
    "exec python3 -m edgeforge worker",
]) + "\\n"
runner_path = base / "run-worker.sh"
runner_path.write_text(runner, encoding="utf-8")
runner_path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
manifest = {{
    "schema_version": 1,
    "edgeforge_version": {VERSION!r},
    "target": profile,
    "source_root": str(source),
    "work_root": str(base / "work"),
    "log_root": str(state / "logs"),
    "backend_advertisement": ["python-reference"],
    "rknn_status": "blocked-until-device-and-correctness" if profile["target"] == "orangepi" else "not-applicable",
    "credentials_written": False,
}}
manifest_path = config / "environment.json"
manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\\n", encoding="utf-8")
print(json.dumps({{"manifest": str(manifest_path), "runner": str(runner_path), "source": str(source), "work": str(base / 'work')}}, sort_keys=True))
'''


def _parse_json(stdout: str) -> dict[str, Any] | None:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def configure_target(*, target: str, source_root: Path, apply: bool, sync_source: bool, timeout: float) -> dict[str, Any]:
    profile = dict(TARGETS[target])
    host = profile["ssh_host"]
    inventory_call = _ssh(host, REMOTE_INVENTORY, timeout)
    inventory = _parse_json(inventory_call.get("stdout", ""))
    record: dict[str, Any] = {
        "schema_version": 1,
        "version": VERSION,
        "target": target,
        "profile": profile,
        "inventory_call": inventory_call,
        "inventory": inventory,
        "status": "ready" if inventory_call.get("status") == "succeeded" and inventory else "blocked",
        "actions": [],
    }
    if inventory and inventory.get("architecture") != profile["architecture"]:
        record["status"] = "blocked"
        record["architecture_error"] = f"expected {profile['architecture']}, got {inventory.get('architecture')}"
    if not apply or record["status"] == "blocked":
        record["digest"] = _digest(record)
        return record
    prepare_call = _ssh(host, _remote_prepare(target, profile), timeout)
    record["actions"].append({"action": "prepare-user-directories", "result": prepare_call})
    if prepare_call.get("status") != "succeeded":
        record["status"] = "blocked"
        record["digest"] = _digest(record)
        return record
    if sync_source:
        sync_call = _rsync(source_root, host, REMOTE_SOURCE, timeout)
        record["actions"].append({"action": "sync-source", "result": sync_call})
        if sync_call.get("status") != "succeeded":
            record["status"] = "blocked"
        else:
            import_call = _ssh(host, "import os,sys; sys.path.insert(0, os.path.expanduser('~/.local/src/edgeforge/src')); import edgeforge; print(edgeforge.__version__)", timeout)
            record["actions"].append({"action": "import-check", "result": import_call})
            record["remote_import_version"] = import_call.get("stdout")
            if import_call.get("status") != "succeeded":
                record["status"] = "blocked"
    record["digest"] = _digest(record)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inventory and prepare user-owned ARM/RISC-V EdgeForge targets")
    parser.add_argument("--target", action="append", choices=sorted(TARGETS), help="target profile; repeat or omit for all")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, default=Path(".edgeforge") / "arm-target-setup" / f"v{VERSION}")
    parser.add_argument("--apply", action="store_true", help="create user-owned target directories and templates")
    parser.add_argument("--sync-source", action="store_true", help="also rsync the source tree, excluding data and logs")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.sync_source and not args.apply:
        parser.error("--sync-source requires --apply")
    targets = args.target or sorted(TARGETS)
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = [configure_target(target=name, source_root=args.source_root, apply=args.apply, sync_source=args.sync_source, timeout=args.timeout) for name in targets]
    for record in records:
        (args.output_root / f"{record['target']}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "schema_version": 1,
        "version": VERSION,
        "mode": "apply" if args.apply else "inventory",
        "sync_source": bool(args.sync_source),
        "records": [{"target": item["target"], "status": item["status"], "digest": item["digest"]} for item in records],
        "status": "succeeded" if all(item["status"] == "ready" for item in records) else "blocked",
    }
    summary["digest"] = _digest(summary)
    (args.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
