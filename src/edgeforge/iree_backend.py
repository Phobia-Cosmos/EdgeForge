"""Trusted prebuilt IREE ukernel backend for EdgeForge pipeline tasks.

The IREE compiler/runtime repository remains a separate source of truth. This
adapter only executes explicitly registered test and benchmark binaries and
turns their results into the normal EdgeForge compile/correctness/benchmark
contract.
"""

from __future__ import annotations

import base64
import json
import math
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from edgeforge import __version__
from edgeforge.operator import OperatorSpec


_REAL_TIME_RE = re.compile(r"real_time\s+([0-9]+(?:\.[0-9]+)?)\s*(ns|us|ms|s)\b")
_THROUGHPUT_RE = re.compile(r"items_per_second=([0-9]+(?:\.[0-9]+)?)([KMG]?)?/s")
_MAX_OUTPUT_CHARS = 1_000_000


def _stage(name: str, status: str, elapsed_ms: float, details: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "status": status, "elapsed_ms": round(elapsed_ms, 3), "details": details}


def _artifact_upload(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    content = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "name": name,
        "kind": "compiler-manifest",
        "media_type": "application/json",
        "metadata": {"backend": "iree-ukernel", "runtime_version": __version__},
        "content_base64": base64.b64encode(content).decode("ascii"),
    }


def _resolve_command(
    value: Any,
    *,
    work_root: Path | None,
    allowed_commands: set[str] | None,
) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise ValueError("IREE kernel metadata command must be a non-empty argv array")
    original = list(value)
    first = original[0]
    if Path(first).is_absolute():
        executable = Path(first).resolve()
    elif work_root is not None:
        executable = (work_root / first).resolve()
        if not executable.is_relative_to(work_root):
            raise ValueError("IREE command escapes the worker work root")
    else:
        resolved = shutil.which(first)
        if not resolved:
            raise RuntimeError(f"IREE executable not found: {first}")
        executable = Path(resolved).resolve()
    if not executable.is_file() or not executable.stat().st_mode & 0o111:
        raise RuntimeError(f"IREE executable is not runnable: {executable}")
    if allowed_commands is not None:
        names = {first, Path(first).name, str(executable), executable.name}
        if not names.intersection(allowed_commands):
            raise RuntimeError(f"IREE command is not allowed on this worker: {first}")
    return [str(executable), *original[1:]]


def _resolve_workdir(value: Any, work_root: Path | None) -> Path | None:
    if value is None:
        return work_root
    if not isinstance(value, str) or not value:
        raise ValueError("IREE kernel metadata workdir must be a non-empty string")
    path = Path(value)
    if not path.is_absolute():
        if work_root is None:
            raise ValueError("relative IREE workdir requires a worker work root")
        path = work_root / path
    path = path.resolve()
    if work_root is not None and not path.is_relative_to(work_root):
        raise ValueError("IREE workdir escapes the worker work root")
    if not path.is_dir():
        raise RuntimeError(f"IREE workdir does not exist: {path}")
    return path


def _set_flag(argv: list[str], name: str, value: Any) -> None:
    prefix = f"--{name}="
    argv[:] = [item for item in argv if not item.startswith(prefix)]
    argv.append(f"{prefix}{value}")


def _benchmark_argv(spec: OperatorSpec, command: list[str], metadata: dict[str, Any]) -> list[str]:
    if len(spec.shape) != 9:
        raise ValueError("IREE conv_nchwc backend requires the packed rank-9 shape")
    n, oc_outer, oh, ow, ic_outer, fh, fw, k0, c0 = spec.shape
    expected_k0 = int(metadata.get("k0", 16))
    expected_c0 = int(metadata.get("c0", 16))
    if (k0, c0) != (expected_k0, expected_c0):
        raise ValueError(f"IREE kernel expects k0={expected_k0}, c0={expected_c0}; got k0={k0}, c0={c0}")
    argv = list(command)
    for name, value in (
        ("n_size", n),
        ("oc_size", oc_outer),
        ("ic_size", ic_outer),
        ("oh_size", oh),
        ("ow_size", ow),
        ("fh_size", fh),
        ("fw_size", fw),
        ("stride_h", spec.attrs["stride_h"]),
        ("stride_w", spec.attrs["stride_w"]),
        ("dilation_h", spec.attrs["dilation_h"]),
        ("dilation_w", spec.attrs["dilation_w"]),
        ("accumulate", str(bool(spec.attrs.get("accumulate", False))).lower()),
    ):
        _set_flag(argv, name, value)
    return argv


def _run(argv: list[str], cwd: Path | None, timeout: float) -> tuple[subprocess.CompletedProcess[str], float]:
    started = time.perf_counter()
    completed = subprocess.run(
        argv,
        cwd=cwd,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
        shell=False,
    )
    return completed, (time.perf_counter() - started) * 1000.0


def _parse_benchmark_output(output: str) -> tuple[list[float], list[float]]:
    timings_ms: list[float] = []
    throughputs: list[float] = []
    scale = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1000.0}
    for match in _REAL_TIME_RE.finditer(output):
        timings_ms.append(float(match.group(1)) * scale[match.group(2)])
    for match in _THROUGHPUT_RE.finditer(output):
        multiplier = {"": 1.0, "K": 1e3, "M": 1e6, "G": 1e9}[match.group(2) or ""]
        throughputs.append(float(match.group(1)) * multiplier)
    return timings_ms, throughputs


def _summary(timings_ms: list[float]) -> dict[str, Any]:
    if not timings_ms:
        return {"runs": 0, "min_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0}
    ordered = sorted(timings_ms)
    p95_index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.95) - 1))
    return {
        "runs": len(timings_ms),
        "min_ms": min(timings_ms),
        "median_ms": statistics.median(timings_ms),
        "p95_ms": ordered[p95_index],
    }


def run_iree_conv_nchwc_pipeline(
    spec: OperatorSpec,
    kernel: dict[str, Any],
    repeats: int,
    *,
    work_root: str | None = None,
    allowed_commands: list[str] | None = None,
) -> dict[str, Any]:
    if spec.name != "conv_nchwc":
        raise ValueError("iree-ukernel backend currently supports only conv_nchwc")
    metadata = kernel.get("metadata") or {}
    if metadata.get("trusted") is not True:
        raise ValueError("iree-ukernel kernels require metadata.trusted=true")
    root = Path(work_root).resolve() if work_root else None
    allowed = set(allowed_commands) if allowed_commands is not None else None
    test_argv = _resolve_command(metadata.get("test_command"), work_root=root, allowed_commands=allowed)
    benchmark_base = _resolve_command(metadata.get("benchmark_command"), work_root=root, allowed_commands=allowed)
    cwd = _resolve_workdir(metadata.get("workdir"), root)
    timeout = min(86_400.0, max(0.1, float(metadata.get("timeout_seconds") or 300.0)))
    repeats = min(100, max(1, int(repeats)))
    pipeline: list[dict[str, Any]] = []
    started = time.perf_counter()
    pipeline.append(
        _stage(
            "compile",
            "succeeded",
            (time.perf_counter() - started) * 1000.0,
            {"mode": "prebuilt", "compiler": metadata.get("compiler", "iree-runtime")},
        )
    )

    test_started = time.perf_counter()
    test_result, test_elapsed = _run(test_argv, cwd, timeout)
    test_output = (test_result.stdout + test_result.stderr)[-_MAX_OUTPUT_CHARS:]
    test_ok = test_result.returncode == 0
    pipeline.append(
        _stage(
            "correctness",
            "succeeded" if test_ok else "failed",
            test_elapsed,
            {"exit_code": test_result.returncode, "output": test_output},
        )
    )
    manifest = {
        "schema_version": 1,
        "backend": "iree-ukernel",
        "runtime_version": __version__,
        "operator": spec.to_dict(),
        "kernel": kernel,
        "source": {
            "repository": metadata.get("iree_repository"),
            "commit": metadata.get("iree_commit"),
            "patch_sha256": metadata.get("patch_sha256"),
        },
    }
    artifact_upload = _artifact_upload(manifest, f"{kernel.get('id', spec.name)}-manifest.json")
    if not test_ok:
        return {
            "operator": spec.to_dict(),
            "backend": "iree-ukernel",
            "kernel_id": kernel.get("id"),
            "kernel_version": kernel.get("version"),
            "correctness": False,
            "errors": [f"correctness executable exited with {test_result.returncode}"],
            "pipeline": pipeline,
            "compile_ms": round(pipeline[0]["elapsed_ms"], 3),
            "elapsed_ms": round((time.perf_counter() - test_started) * 1000.0, 3),
            "exit_code": 1,
            "artifact_upload": artifact_upload,
        }

    benchmark_timings: list[float] = []
    throughputs: list[float] = []
    benchmark_outputs: list[str] = []
    benchmark_started = time.perf_counter()
    benchmark_ok = True
    for _ in range(repeats):
        benchmark_argv = _benchmark_argv(spec, benchmark_base, metadata)
        result, elapsed = _run(benchmark_argv, cwd, timeout)
        output = (result.stdout + result.stderr)[-_MAX_OUTPUT_CHARS:]
        benchmark_outputs.append(output)
        parsed_timings, parsed_throughputs = _parse_benchmark_output(output)
        benchmark_timings.extend(parsed_timings or [elapsed])
        throughputs.extend(parsed_throughputs)
        if result.returncode != 0:
            benchmark_ok = False
            break
    benchmark_elapsed = (time.perf_counter() - benchmark_started) * 1000.0
    pipeline.append(
        _stage(
            "benchmark",
            "succeeded" if benchmark_ok else "failed",
            benchmark_elapsed,
            {"summary": _summary(benchmark_timings), "throughput_items_per_second": throughputs},
        )
    )
    return {
        "operator": spec.to_dict(),
        "backend": "iree-ukernel",
        "kernel_id": kernel.get("id"),
        "kernel_version": kernel.get("version"),
        "correctness": test_ok and benchmark_ok,
        "errors": [] if benchmark_ok else ["benchmark executable failed"],
        "timings_ms": benchmark_timings,
        "summary": _summary(benchmark_timings),
        "throughput_items_per_second": throughputs,
        "raw_outputs": benchmark_outputs[-3:],
        "pipeline": pipeline,
        "compile_ms": round(pipeline[0]["elapsed_ms"], 3),
        "elapsed_ms": round(sum(stage["elapsed_ms"] for stage in pipeline), 3),
        "exit_code": 0 if test_ok and benchmark_ok else 1,
        "artifact_upload": artifact_upload,
    }
