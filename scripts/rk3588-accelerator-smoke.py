#!/usr/bin/env python3
"""Run and persist an offline RK3588 GPU/NPU runtime smoke.

The remote mode sends this file to ``python3 -`` over SSH, so it does not
require a system installation or a camera on the board.  Only the small
``edgeforge.accelerator_probe`` module is synchronized to a user-owned cache.
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


VERSION = "0.16.1"


def _load_probe_module():
    try:
        from edgeforge.accelerator_probe import result_digest, run_accelerator_probe

        return run_accelerator_probe, result_digest
    except ModuleNotFoundError:
        # When the script is executed directly from a source checkout without
        # installing EdgeForge, make the adjacent ``src`` tree importable.
        source_root = Path(__file__).resolve().parents[1] / "src"
        sys.path.insert(0, str(source_root))
        from edgeforge.accelerator_probe import result_digest, run_accelerator_probe

        return run_accelerator_probe, result_digest


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _remote_home(host: str, timeout: float) -> str:
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, "printf", "%s", "$HOME"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise RuntimeError(f"cannot determine remote home for {host}: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _remote_probe(args: argparse.Namespace) -> tuple[dict[str, Any], str, str]:
    """Upload only probe source and execute it on the target as a user."""

    host = args.ssh_host
    remote_home = _remote_home(host, args.timeout_seconds)
    remote_root = Path(remote_home) / ".cache" / "edgeforge" / "accelerator-smoke" / f"v{args.version}"
    remote_source = remote_root / "src" / "edgeforge"
    remote_runner = remote_root / "runner.py"
    mkdir = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, "mkdir", "-p", str(remote_source)],
        capture_output=True,
        text=True,
        timeout=args.timeout_seconds,
        check=False,
    )
    if mkdir.returncode != 0:
        raise RuntimeError(f"remote cache setup failed: {mkdir.stderr.strip()}")
    source_file = Path(__file__).resolve().parents[1] / "src" / "edgeforge" / "accelerator_probe.py"
    sync = subprocess.run(
        ["rsync", "-a", str(source_file), f"{host}:{remote_source}/accelerator_probe.py"],
        capture_output=True,
        text=True,
        timeout=args.timeout_seconds,
        check=False,
    )
    if sync.returncode != 0:
        raise RuntimeError(f"probe source sync failed: {sync.stderr.strip()}")

    source_text = Path(__file__).read_text(encoding="utf-8")
    remote_command = [
        "python3",
        "-",
        "--local",
        "--name",
        args.name,
        "--version",
        args.version,
        "--repeat",
        str(args.repeat),
        "--output",
        "-",
        "--probe-source-root",
        str(remote_root / "src"),
    ]
    if args.model:
        remote_command.extend(["--model", args.model])
    if args.runtime_library:
        remote_command.extend(["--runtime-library", args.runtime_library])
    for flag, enabled in (("--skip-opencl", args.skip_opencl), ("--skip-vulkan", args.skip_vulkan), ("--skip-rknn", args.skip_rknn)):
        if enabled:
            remote_command.append(flag)
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, *remote_command],
        input=source_text,
        capture_output=True,
        text=True,
        timeout=args.timeout_seconds,
        check=False,
    )
    remote_diagnostics = completed.stderr.strip()
    try:
        # Vendor OpenCL/RKNPU libraries occasionally print a banner to stdout
        # after the JSON has been flushed.  Decode the first complete object
        # and retain the trailing bytes as diagnostic evidence.
        decoder = json.JSONDecoder()
        stripped = completed.stdout.lstrip()
        value, end = decoder.raw_decode(stripped)
        trailing = stripped[end:].strip()
        if trailing:
            remote_diagnostics = (remote_diagnostics + "\n" + trailing).strip()
    except json.JSONDecodeError as error:
        raise RuntimeError(f"remote probe returned non-JSON output: {completed.stdout[:500]!r}; stderr={completed.stderr[:500]!r}") from error
    # The board probe intentionally returns 2 for a structured partial or
    # blocked capability result.  That is evidence, not transport failure.
    if completed.returncode not in {0, 1, 2}:
        raise RuntimeError(f"remote accelerator smoke transport failed (exit {completed.returncode}): {completed.stderr.strip()}")
    if not isinstance(value, dict):
        raise RuntimeError("remote probe returned a non-object JSON value")
    return value, completed.stdout, remote_diagnostics


def _run_local(args: argparse.Namespace) -> dict[str, Any]:
    run_probe, _ = _load_probe_module()
    return run_probe(
        name=args.name,
        model_path=args.model,
        runtime_library=args.runtime_library,
        repeat=args.repeat,
        skip_opencl=args.skip_opencl,
        skip_vulkan=args.skip_vulkan,
        skip_rknn=args.skip_rknn,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline RK3588 Mali/RKNPU2 runtime smoke")
    parser.add_argument("--name", default="orangepi")
    parser.add_argument("--ssh-host", help="run the probe on a target over SSH")
    parser.add_argument("--local", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--probe-source-root", help=argparse.SUPPRESS)
    parser.add_argument("--model", help="board-local .rknn model; defaults to a user-cache model, then vendor demo")
    parser.add_argument("--runtime-library", help="board-local librknnrt.so/librknn_api.so")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--skip-opencl", action="store_true")
    parser.add_argument("--skip-vulkan", action="store_true")
    parser.add_argument("--skip-rknn", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    args = parser.parse_args(argv)

    # The remote cache contains the module, while stdin contains this runner.
    # Add that source root before importing in ``--local`` mode.
    if args.probe_source_root:
        sys.path.insert(0, args.probe_source_root)

    remote_stdout = ""
    remote_stderr = ""
    started = time.perf_counter()
    try:
        if args.ssh_host and not args.local:
            result, remote_stdout, remote_stderr = _remote_probe(args)
        else:
            result = _run_local(args)
    except Exception as error:
        result = {
            "schema_version": 1,
            "name": args.name,
            "status": "offline",
            "probe_mode": "offline-api-smoke",
            "error": str(error),
            "scientific_conclusion_allowed": False,
        }

    digest = hashlib.sha256(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    result["result_digest"] = digest
    result["runner"] = {"version": args.version, "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3)}
    args.output.parent.mkdir(parents=True, exist_ok=True) if str(args.output) != "-" else None
    if str(args.output) != "-":
        args.output.write_text(_json(result), encoding="utf-8")

    log_path = args.log_dir / f"v{args.version}" / "rk3588-accelerator-smoke" / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{args.name}-{digest[:8]}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_record = {
        "timestamp": time.time(),
        "version": args.version,
        "component": "rk3588-accelerator-smoke",
        "target": args.name,
        "ssh_host": args.ssh_host,
        "command": " ".join(shlex.quote(item) for item in sys.argv),
        "result_digest": digest,
        "result": result,
        "remote_stderr": remote_stderr,
    }
    log_path.write_text(json.dumps(log_record, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(_json(result), end="")
    return 0 if result.get("status") == "pass" else 2 if result.get("status") in {"partial", "blocked"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
