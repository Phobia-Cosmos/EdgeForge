"""Capability-only deployment preflight; never executes model code."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _required_capabilities(backend: str) -> list[str]:
    return {
        "python-reference": [],
        "torch-eager": ["torch_python"],
        "torch-compile": ["torch_python"],
        "onnx-runtime": ["onnxruntime_python"],
        "iree": ["iree_tools"],
        # RKNN-Toolkit2 is a host-side conversion package.  The board-side
        # runtime is the C ``librknnrt.so``/``librknn_api.so`` pair, and recent
        # RK3588 kernels expose the NPU through DRM rather than /dev/rknpu*.
        "rknn": ["rknn_runtime_files"],
        "opencl": ["opencl_userspace"],
        "vulkan": ["vulkan_loader", "vulkan_icd_manifest"],
    }.get(backend, [])


def _alternative_capabilities(backend: str) -> list[list[str]]:
    """Return OR-groups, each of which must have at least one capability."""

    if backend == "rknn":
        return [["rk3588_npu_drm", "rk3588_npu_device", "rk3588_npu_platform"]]
    return []


def evaluate_preflight(manifest: dict[str, Any], probe: dict[str, Any]) -> dict[str, Any]:
    target = manifest.get("target") or {}
    compiler = manifest.get("compiler") or {}
    backend = str(compiler.get("backend") or "")
    expected_architecture = str(target.get("architecture") or "")
    actual_architecture = str((probe.get("summary") or {}).get("architecture") or "")
    available = probe.get("runtime_capabilities") or {}
    reasons: list[str] = []
    if not backend:
        reasons.append("manifest.compiler.backend is missing")
    if expected_architecture and actual_architecture and expected_architecture != actual_architecture:
        reasons.append(f"target architecture mismatch: manifest={expected_architecture}, probe={actual_architecture}")
    if probe.get("status") in {"offline", "unreachable"}:
        reasons.append(f"target probe status is {probe.get('status')}")
    required = _required_capabilities(backend)
    missing = [name for name in required if not bool(available.get(name))]
    alternatives = _alternative_capabilities(backend)
    missing_alternatives: list[list[str]] = []
    for group in alternatives:
        if not any(bool(available.get(name)) for name in group):
            missing_alternatives.append(group)
    if missing:
        reasons.append("missing runtime capabilities: " + ", ".join(missing))
    if missing_alternatives:
        reasons.append("missing one of runtime capabilities: " + "; ".join("|".join(group) for group in missing_alternatives))
    return {
        "schema_version": 1,
        "status": "PASS" if not reasons else "BLOCKED",
        "manifest_model": (manifest.get("model") or {}).get("name"),
        "backend": backend,
        "target": target,
        "probe_target": probe.get("name"),
        "probe_status": probe.get("status"),
        "required_capabilities": required,
        "missing_capabilities": missing,
        "alternative_capabilities": alternatives,
        "missing_alternative_capabilities": missing_alternatives,
        "reasons": reasons,
        "manifest_digest": hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "probe_digest": hashlib.sha256(json.dumps(probe, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "execution_performed": False,
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value
