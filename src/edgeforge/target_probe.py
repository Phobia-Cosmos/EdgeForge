"""Read-only target capability probing for heterogeneous EdgeForge workers."""

from __future__ import annotations

import hashlib
import json
import platform
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


PROBES: dict[str, list[str]] = {
    "architecture": ["uname", "-m"],
    "kernel": ["uname", "-srvm"],
    "cpu_count": ["nproc"],
    "cpu_info": ["lscpu"],
    "memory_info": ["cat", "/proc/meminfo"],
    "vulkan_info": [
        "sh",
        "-lc",
        "candidate=$(command -v vulkaninfo || true); if test -z \"$candidate\" && test -x \"$HOME/.cache/edgeforge/rk3588-smoke/vulkan-tools/root/usr/bin/vulkaninfo\"; then candidate=\"$HOME/.cache/edgeforge/rk3588-smoke/vulkan-tools/root/usr/bin/vulkaninfo\"; fi; if test -n \"$candidate\"; then timeout -k 1 8s \"$candidate\" --summary; else echo unavailable; fi",
    ],
    "iree_tools": ["sh", "-lc", "for x in iree-compile iree-run-module; do if command -v \"$x\" >/dev/null 2>&1; then \"$x\" --version 2>&1 | head -1; else echo \"$x: unavailable\"; fi; done"],
    "python_runtimes": ["python3", "-c", "import importlib.util,sys; print('python='+sys.version.split()[0]); print('numpy='+str(bool(importlib.util.find_spec('numpy')))); print('torch='+str(bool(importlib.util.find_spec('torch')))); print('onnxruntime='+str(bool(importlib.util.find_spec('onnxruntime')))); print('iree='+str(bool(importlib.util.find_spec('iree')))); print('rknn='+str(bool(importlib.util.find_spec('rknn'))))"],
    "nvidia_gpu": ["sh", "-lc", "if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; else echo unavailable; fi"],
    "rknpu_device": ["sh", "-lc", "if find /dev -maxdepth 1 -name 'rknpu*' -print -quit 2>/dev/null | grep -q .; then find /dev -maxdepth 1 -name 'rknpu*' -print | sort; else echo absent; fi"],
    "rknn_runtime": ["sh", "-lc", "for path in /usr/bin/rknn_server /usr/bin/rknn_demo /usr/lib/librknnrt.so /usr/lib/librknn_api.so; do if test -e \"$path\"; then echo \"$path\"; fi; done"],
    "drm_devices": ["sh", "-lc", "if test -d /dev/dri; then find /dev/dri -maxdepth 1 -type c -printf '%f\\n' | sort; else echo absent; fi"],
    # Recent RK3588 kernels expose RKNPU as a DRM render node (for example
    # renderD129), so a /dev/rknpu* character node is not required.
    "rknpu_drm_driver": ["sh", "-lc", "found=0; for link in /sys/class/drm/card*/device/driver /sys/class/drm/renderD*/device/driver; do if test -e \"$link\"; then found=1; printf '%s=%s\\n' \"$link\" \"$(basename \"$(readlink -f \"$link\")\")\"; fi; done; test $found -eq 1 || echo absent"],
    "rknpu_platform": ["sh", "-lc", "if test -d /sys/bus/platform/devices; then find /sys/bus/platform/devices -maxdepth 1 -iname '*npu*' -printf '%f\\n' | sort; else echo absent; fi"],
    "gpu_userspace": ["sh", "-lc", "{ ldconfig -p 2>/dev/null | grep -E 'lib(OpenCL|vulkan|mali)' || true; for path in /etc/OpenCL/vendors/*.icd /etc/vulkan/icd.d/*.json /usr/share/vulkan/icd.d/*.json; do for item in $path; do if test -e \"$item\"; then echo \"$item\"; fi; done; done; } | sort -u"],
    "opencl_info": ["sh", "-lc", "if command -v clinfo >/dev/null 2>&1; then timeout -k 1 12s clinfo 2>&1 | grep -E 'Platform Name|Platform Version|Device Name|Device Type|Device Available|Device Version|Driver Version|Max compute units' | head -80; else echo clinfo: unavailable; fi"],
    "vulkan_loader": ["sh", "-lc", "if ldconfig -p 2>/dev/null | grep -q 'libvulkan.so.1'; then ldconfig -p 2>/dev/null | grep 'libvulkan.so.1' | head -5; else echo unavailable; fi"],
}


def _run(argv: list[str], *, timeout_seconds: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_seconds, check=False)
        status = "succeeded" if completed.returncode == 0 else "unavailable"
        return {
            "status": status,
            "exit_code": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "status": "unreachable" if isinstance(error, subprocess.TimeoutExpired) else "unavailable",
            "exit_code": None,
            "stdout": "",
            "stderr": str(error),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }


def _parse_mem_total(meminfo: str) -> int | None:
    for line in meminfo.splitlines():
        if line.startswith("MemTotal:"):
            fields = line.split()
            if len(fields) >= 2:
                try:
                    return int(fields[1]) // 1024
                except ValueError:
                    return None
    return None


def _parse_cpu_model(cpu_info: str) -> str | None:
    for line in cpu_info.splitlines():
        if ":" in line and line.split(":", 1)[0].strip().lower() in {"model name", "model", "hardware", "cpu implementer"}:
            value = line.split(":", 1)[1].strip()
            if value:
                return value
    return None


def probe_target(*, name: str, ssh_host: str | None = None, timeout_seconds: float = 8.0) -> dict[str, Any]:
    """Probe local or SSH target and return JSON-safe capability evidence."""
    results: dict[str, dict[str, Any]] = {}
    for key, command in PROBES.items():
        argv = list(command)
        if ssh_host:
            # SSH concatenates the remote arguments into one command string;
            # quote the probe argv so `sh -lc` snippets remain one unit.
            argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", ssh_host, shlex.join(argv)]
        results[key] = _run(argv, timeout_seconds=timeout_seconds)

    architecture = results["architecture"].get("stdout") or "unknown"
    cpu_info = results["cpu_info"].get("stdout") or ""
    memory_info = results["memory_info"].get("stdout") or ""
    available = [key for key, result in results.items() if result.get("status") == "succeeded"]
    fingerprint_payload = {
        "architecture": architecture,
        "cpu_model": _parse_cpu_model(cpu_info),
        "cpu_count": results["cpu_count"].get("stdout"),
        "memory_total_mb": _parse_mem_total(memory_info),
        "vulkan": "unavailable" not in (results["vulkan_info"].get("stdout") or ""),
        "rknpu_char": bool((results["rknpu_device"].get("stdout") or "").strip()) and (results["rknpu_device"].get("stdout") or "").strip() != "absent",
        "rknpu_drm_driver": results["rknpu_drm_driver"].get("stdout"),
    }
    fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]
    vulkan_output = results["vulkan_info"].get("stdout") or ""
    iree_output = results["iree_tools"].get("stdout") or ""
    python_output = results["python_runtimes"].get("stdout") or ""
    rknn_runtime_output = results["rknn_runtime"].get("stdout") or ""
    drm_output = results["drm_devices"].get("stdout") or ""
    rknpu_driver_output = results["rknpu_drm_driver"].get("stdout") or ""
    rknpu_platform_output = results["rknpu_platform"].get("stdout") or ""
    gpu_userspace_output = results["gpu_userspace"].get("stdout") or ""
    opencl_output = results["opencl_info"].get("stdout") or ""
    vulkan_loader_output = results["vulkan_loader"].get("stdout") or ""
    has_rknpu_char = bool((results["rknpu_device"].get("stdout") or "").strip()) and (results["rknpu_device"].get("stdout") or "").strip() != "absent"
    has_rknpu_drm = any("=RKNPU" in line.upper() for line in rknpu_driver_output.splitlines())
    has_rknpu_platform = bool(rknpu_platform_output.strip()) and rknpu_platform_output.strip() != "absent"
    has_opencl_userspace = any("opencl" in line.lower() or "mali" in line.lower() for line in gpu_userspace_output.splitlines())
    has_vulkan_loader = "libvulkan.so.1" in vulkan_loader_output
    return {
        "schema_version": 1,
        "name": name,
        "probe_scope": "ssh" if ssh_host else "local",
        "ssh_host": ssh_host,
        "status": "online" if len(available) == len(PROBES) else ("partial" if available else "offline"),
        "probed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "probe_runner": {"python": platform.python_version(), "platform": platform.platform()},
        "summary": {
            "architecture": architecture,
            "cpu_model": _parse_cpu_model(cpu_info),
            "cpu_count": results["cpu_count"].get("stdout"),
            "memory_total_mb": _parse_mem_total(memory_info),
            "hardware_fingerprint": fingerprint,
        },
        "runtime_capabilities": {
            "vulkan_info": results["vulkan_info"].get("status") == "succeeded" and "unavailable" not in vulkan_output and "absent" not in vulkan_output,
            "numpy_python": "numpy=True" in python_output,
            "torch_python": "torch=True" in python_output,
            "onnxruntime_python": "onnxruntime=True" in python_output,
            "iree_tools": "unavailable" not in iree_output,
            "iree_python": "iree=True" in python_output,
            "rknn_python": "rknn=True" in python_output,
            # Keep the character-node field for compatibility; modern RK3588
            # images may expose the same accelerator only through DRM.
            "rk3588_npu_device": has_rknpu_char,
            "rk3588_npu_drm": has_rknpu_drm,
            "rk3588_npu_platform": has_rknpu_platform,
            "rknn_runtime_files": bool(rknn_runtime_output.strip()) and "unavailable" not in rknn_runtime_output,
            "drm": bool(drm_output and "absent" not in drm_output),
            "opencl_userspace": has_opencl_userspace,
            "opencl_device_evidence": "Device Name" in opencl_output and "Mali" in opencl_output,
            "vulkan_loader": has_vulkan_loader,
            "vulkan_icd_manifest": any(".json" in line and "vulkan" in line.lower() for line in gpu_userspace_output.splitlines()),
        },
        "probes": results,
    }


def write_probe(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
