"""Explicit model backend and target capability contract.

The registry is intentionally small and descriptive.  It does not import a
framework or claim that a worker can execute a backend; worker advertisement
remains the source of runtime capability truth.
"""

from __future__ import annotations

from typing import Any


BACKENDS: dict[str, dict[str, Any]] = {
    "python-reference": {
        "kind": "reference",
        "architectures": ["*"],
        "accelerators": [],
        "requires_explicit_target": False,
    },
    "torch-eager": {
        "kind": "framework-runtime",
        "architectures": ["x86_64", "aarch64"],
        "accelerators": [],
        "requires_explicit_target": True,
    },
    "torch-compile": {
        "kind": "compiler-runtime",
        "architectures": ["x86_64", "aarch64"],
        "accelerators": [],
        "requires_explicit_target": True,
    },
    "onnx-runtime": {
        "kind": "portable-runtime",
        "architectures": ["x86_64", "aarch64", "riscv64"],
        "accelerators": [],
        "requires_explicit_target": True,
    },
    "triton": {
        "kind": "gpu-kernel-runtime",
        "architectures": ["x86_64"],
        "accelerators": ["nvidia-gpu"],
        "requires_explicit_target": True,
    },
    "iree": {
        "kind": "compiler-runtime",
        "architectures": ["x86_64", "aarch64", "riscv64"],
        "accelerators": [],
        "requires_explicit_target": True,
        "explicit_target_fields": ["architecture", "device"],
    },
    "rknn": {
        "kind": "npu-runtime",
        "architectures": ["aarch64"],
        "accelerators": ["rk3588-npu"],
        "requires_explicit_target": True,
    },
}


def backend_capabilities() -> list[dict[str, Any]]:
    return [{"name": name, **dict(spec)} for name, spec in sorted(BACKENDS.items())]


def validate_backend_target(backend: str, target: dict[str, Any]) -> dict[str, Any]:
    backend = str(backend or "").strip()
    if not backend:
        raise ValueError("compiler.backend is required")
    if not isinstance(target, dict):
        raise ValueError("target must be an object")
    spec = BACKENDS.get(backend)
    if spec is None:
        # Custom backends remain possible, but must opt into explicit target
        # declaration rather than silently inheriting a host default.
        if not str(target.get("architecture") or "").strip():
            raise ValueError(f"unknown backend {backend!r} requires target.architecture")
        return {"name": backend, "kind": "custom", "architectures": [target["architecture"]], "accelerators": []}
    if spec.get("requires_explicit_target") and not str(target.get("architecture") or "").strip():
        raise ValueError(f"backend {backend} requires explicit target.architecture")
    for field in spec.get("explicit_target_fields") or []:
        if not str(target.get(field) or "").strip():
            raise ValueError(f"backend {backend} requires explicit target.{field}")
    architecture = str(target.get("architecture") or "")
    if architecture and "*" not in spec["architectures"] and architecture not in spec["architectures"]:
        raise ValueError(f"backend {backend} does not support target architecture {architecture}")
    accelerator = str(target.get("accelerator") or "")
    required_accelerators = set(spec.get("accelerators") or [])
    if required_accelerators and accelerator not in required_accelerators:
        raise ValueError(f"backend {backend} requires target accelerator in {sorted(required_accelerators)}")
    return {"name": backend, **dict(spec)}
