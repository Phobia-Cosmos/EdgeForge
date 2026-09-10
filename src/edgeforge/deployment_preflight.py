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


def _runtime_validation_status(backend: str, validation: dict[str, Any] | None) -> tuple[bool, str]:
    """Check an already-recorded accelerator API smoke; never execute code."""

    if backend not in {"rknn", "opencl", "vulkan"}:
        return True, "not-required"
    if not validation:
        return False, "missing"
    if backend == "rknn":
        npu = validation.get("npu") or {}
        step_values = {
            "init": npu.get("init"),
            "sdk": (npu.get("sdk") or {}).get("result") or (npu.get("sdk") or {}),
            "io_query": npu.get("io_query"),
            "inputs_set": npu.get("inputs_set"),
            "destroy": npu.get("destroy"),
        }
        steps_ok = all((step_values.get(step) or {}).get("code") == 0 for step in step_values)
        passed = npu.get("status") == "pass" and steps_ok and bool(npu.get("deterministic_zero_input"))
        return passed, "pass" if passed else "npu smoke did not pass init/query/input/run/output/determinism checks"
    section = (validation.get("gpu") or {}).get(backend) or {}
    passed = section.get("status") == "pass"
    return passed, "pass" if passed else f"{backend} smoke did not pass"


def evaluate_preflight(
    manifest: dict[str, Any],
    probe: dict[str, Any],
    runtime_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = manifest.get("target") or {}
    compiler = manifest.get("compiler") or {}
    backend = str(compiler.get("backend") or "")
    expected_architecture = str(target.get("architecture") or "")
    actual_architecture = str((probe.get("summary") or {}).get("architecture") or "")
    available = probe.get("runtime_capabilities") or {}
    validation = runtime_validation if runtime_validation is not None else probe.get("runtime_validation")
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
    validation_pass, validation_reason = _runtime_validation_status(backend, validation)
    if validation and validation.get("name") and probe.get("name") and validation.get("name") != probe.get("name"):
        validation_pass = False
        validation_reason = f"validation target mismatch: validation={validation.get('name')}, probe={probe.get('name')}"
    if backend in {"rknn", "opencl", "vulkan"} and not validation_pass:
        reasons.append("missing successful runtime API validation: " + validation_reason)
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
        "runtime_validation_required": backend in {"rknn", "opencl", "vulkan"},
        "runtime_validation_status": "PASS" if validation_pass else "BLOCKED",
        "runtime_validation_reason": validation_reason,
        "runtime_validation_digest": hashlib.sha256(json.dumps(validation, sort_keys=True, separators=(",", ":")).encode()).hexdigest() if validation is not None else None,
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
