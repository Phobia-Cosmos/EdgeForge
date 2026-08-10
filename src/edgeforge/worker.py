"""Portable worker runtime for x86_64, ARM64, and RISC-V64 nodes."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import platform
import shutil
import socket
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from edgeforge import __version__
from edgeforge.client import APIError, Client
from edgeforge.compiler import run_kernel_pipeline
from edgeforge.operator import OperatorSpec, benchmark_operator


LOG = logging.getLogger("edgeforge.worker")
MAX_OUTPUT_CHARS = 1_000_000


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text(errors="replace").strip()
    except OSError:
        return ""


def _architecture() -> str:
    machine = platform.machine().lower()
    aliases = {"amd64": "x86_64", "arm64": "aarch64", "riscv": "riscv64"}
    return aliases.get(machine, machine)


def _memory_info() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in _read_text("/proc/meminfo").splitlines():
        key, _, value = line.partition(":")
        if key in {"MemTotal", "MemAvailable"}:
            try:
                values[key] = int(value.strip().split()[0]) // 1024
            except (IndexError, ValueError):
                pass
    return values


def _cpu_model() -> str:
    for line in _read_text("/proc/cpuinfo").splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() in {"model name", "hardware", "uarch"} and value.strip():
            return value.strip()
    return platform.processor() or "unknown"


def _accelerators() -> list[str]:
    accelerators: list[str] = []
    if shutil.which("nvidia-smi"):
        accelerators.append("nvidia-gpu")
    compatible = _read_text("/proc/device-tree/compatible").replace("\x00", ",").lower()
    if Path("/dev/rknpu").exists() or "rk3588" in compatible:
        accelerators.append("rk3588-npu")
    if Path("/dev/dri").exists():
        accelerators.append("drm")
    return accelerators


def collect_capabilities() -> dict[str, Any]:
    memory = _memory_info()
    runtimes = [name for name in ("python3", "git", "cmake", "ninja", "gcc", "clang", "nvidia-smi") if shutil.which(name)]
    capabilities = {
        "architecture": _architecture(),
        "os": platform.system().lower(),
        "kernel": platform.release(),
        "cpu_count": os.cpu_count() or 1,
        "cpu_model": _cpu_model(),
        "memory_total_mb": memory.get("MemTotal", 0),
        "accelerators": _accelerators(),
        "runtimes": runtimes,
        "python": platform.python_version(),
    }
    identity = {
        "architecture": capabilities["architecture"],
        "cpu_model": capabilities["cpu_model"],
        "cpu_count": capabilities["cpu_count"],
        "memory_total_mb": capabilities["memory_total_mb"],
        "accelerators": capabilities["accelerators"],
    }
    capabilities["hardware_fingerprint"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    return capabilities


def _temperatures() -> dict[str, float]:
    values = []
    for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            value = float(path.read_text().strip())
            if value > 1000:
                value /= 1000.0
            if -20 <= value <= 150:
                values.append(value)
        except (OSError, ValueError):
            continue
    if not values:
        return {}
    return {"temperature_avg_c": round(statistics.fmean(values), 1), "temperature_max_c": round(max(values), 1)}


def collect_metrics(work_root: Path) -> dict[str, Any]:
    memory = _memory_info()
    try:
        load_1m, load_5m, load_15m = os.getloadavg()
    except OSError:
        load_1m = load_5m = load_15m = 0.0
    disk = shutil.disk_usage(work_root)
    metrics: dict[str, Any] = {
        "load_1m": round(load_1m, 3),
        "load_5m": round(load_5m, 3),
        "load_15m": round(load_15m, 3),
        "memory_available_mb": memory.get("MemAvailable", 0),
        "disk_available_mb": disk.free // (1024 * 1024),
    }
    metrics.update(_temperatures())
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
            )
            gpu, used, total = [int(part.strip()) for part in result.stdout.splitlines()[0].split(",")]
            metrics.update({"gpu_utilization_percent": gpu, "gpu_memory_used_mb": used, "gpu_memory_total_mb": total})
        except (IndexError, OSError, subprocess.SubprocessError, ValueError):
            pass
    return metrics


def default_worker_id() -> str:
    machine_id = _read_text("/etc/machine-id") or socket.gethostname()
    suffix = hashlib.sha256(machine_id.encode()).hexdigest()[:8]
    return f"{socket.gethostname()}-{suffix}"


class Worker:
    def __init__(
        self,
        client: Client,
        worker_id: str,
        labels: dict[str, str],
        work_root: Path,
        interval: float,
        allowed_commands: set[str],
        allow_any_command: bool,
    ):
        self.client = client
        self.worker_id = worker_id
        self.labels = labels
        self.work_root = work_root.resolve()
        self.interval = interval
        self.allowed_commands = allowed_commands
        self.allow_any_command = allow_any_command
        self.stop_event = threading.Event()
        self.work_root.mkdir(parents=True, exist_ok=True)

    def registration(self) -> dict[str, Any]:
        return {
            "id": self.worker_id,
            "hostname": socket.gethostname(),
            "capabilities": collect_capabilities(),
            "metrics": collect_metrics(self.work_root),
            "labels": self.labels,
            "version": __version__,
        }

    def register(self) -> None:
        self.client.request("POST", "/api/v1/workers/register", self.registration())
        LOG.info("registered worker %s", self.worker_id)

    def heartbeat(self) -> None:
        self.client.request(
            "POST",
            f"/api/v1/workers/{self.worker_id}/heartbeat",
            {"metrics": collect_metrics(self.work_root)},
        )

    def _heartbeat_loop(self) -> None:
        while not self.stop_event.wait(self.interval):
            try:
                self.heartbeat()
            except APIError as error:
                LOG.warning("heartbeat failed: %s", error)
                try:
                    self.register()
                except APIError:
                    pass

    def _resolve_command(self, command: str) -> str:
        executable = command if os.path.isabs(command) else shutil.which(command)
        if not executable:
            raise RuntimeError(f"executable not found: {command}")
        resolved = str(Path(executable).resolve())
        names = {command, Path(command).name, resolved, Path(resolved).name}
        if not self.allow_any_command and not names.intersection(self.allowed_commands):
            raise RuntimeError(f"command is not allowed on this worker: {command}")
        return resolved

    def _resolve_cwd(self, value: str | None) -> Path:
        path = (self.work_root / value).resolve() if value else self.work_root
        if not path.is_relative_to(self.work_root):
            raise RuntimeError("task cwd escapes the configured work root")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def execute(self, task: dict[str, Any]) -> dict[str, Any]:
        payload = task["payload"]
        if task["kind"] == "kernel_pipeline":
            return run_kernel_pipeline(payload)
        if task["kind"] == "operator_benchmark":
            spec = OperatorSpec.from_payload(payload.get("operator") or {})
            result = benchmark_operator(spec, int(payload.get("repeats") or 3))
            kernel = payload.get("kernel") or {}
            if kernel:
                result["kernel_id"] = kernel.get("id")
                result["kernel_version"] = kernel.get("version")
            result["exit_code"] = 0 if result["correctness"] else 1
            result["elapsed_ms"] = round(sum(result.get("timings_ms") or []), 3)
            return result
        argv = list(payload["argv"])
        argv[0] = self._resolve_command(argv[0])
        cwd = self._resolve_cwd(payload.get("cwd"))
        timeout = min(86_400.0, max(0.1, float(payload.get("timeout_seconds") or 300.0)))
        repeats = int(payload.get("repeats") or 1) if task["kind"] == "benchmark" else 1
        repeats = min(100, max(1, repeats))
        env_overrides = payload.get("env") or {}
        if not isinstance(env_overrides, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env_overrides.items()):
            raise RuntimeError("payload.env must contain string keys and values")
        env = os.environ.copy()
        env.update(env_overrides)

        timings = []
        last_result: subprocess.CompletedProcess[str] | None = None
        started = time.perf_counter()
        for _ in range(repeats):
            run_started = time.perf_counter()
            last_result = subprocess.run(
                argv,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
                shell=False,
            )
            timings.append(round((time.perf_counter() - run_started) * 1000.0, 3))
            if last_result.returncode != 0:
                break
        assert last_result is not None
        result: dict[str, Any] = {
            "argv": argv,
            "cwd": str(cwd),
            "exit_code": last_result.returncode,
            "stdout": last_result.stdout[-MAX_OUTPUT_CHARS:],
            "stderr": last_result.stderr[-MAX_OUTPUT_CHARS:],
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "timings_ms": timings,
        }
        if timings:
            ordered = sorted(timings)
            p95_index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.95) - 1))
            result["summary"] = {
                "runs": len(timings),
                "min_ms": min(timings),
                "median_ms": statistics.median(timings),
                "p95_ms": ordered[p95_index],
            }
        return result

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.register()
                break
            except APIError as error:
                LOG.warning("registration failed: %s", error)
                self.stop_event.wait(self.interval)
        thread = threading.Thread(target=self._heartbeat_loop, name="heartbeat", daemon=True)
        thread.start()
        try:
            while not self.stop_event.is_set():
                try:
                    response = self.client.request("POST", f"/api/v1/workers/{self.worker_id}/lease", {})
                    task = response.get("task")
                    if task is None:
                        self.stop_event.wait(self.interval)
                        continue
                    LOG.info("running task %s (%s)", task["id"], task["kind"])
                    try:
                        result = self.execute(task)
                        status = "succeeded" if result["exit_code"] == 0 else "failed"
                        completion = {
                            "worker_id": self.worker_id,
                            "runtime_version": __version__,
                            "status": status,
                            "result": result,
                        }
                    except Exception as error:
                        LOG.exception("task %s failed", task["id"])
                        completion = {
                            "worker_id": self.worker_id,
                            "runtime_version": __version__,
                            "status": "failed",
                            "error": str(error),
                            "result": {},
                        }
                    self.client.request("POST", f"/api/v1/tasks/{task['id']}/complete", completion)
                except APIError as error:
                    LOG.warning("worker loop error: %s", error)
                    self.stop_event.wait(self.interval)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_event.set()
            thread.join(timeout=self.interval + 1)
