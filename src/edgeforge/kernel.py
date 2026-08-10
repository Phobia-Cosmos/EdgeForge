"""Kernel Registry records and compatibility predicates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class KernelSpec:
    id: str
    operator: str
    backend: str
    version: str
    architectures: tuple[str, ...] = ("*",)
    accelerators: tuple[str, ...] = ()
    dtypes: tuple[str, ...] = ("fp32",)
    shape_constraints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "KernelSpec":
        if not isinstance(payload, dict):
            raise ValueError("kernel must be an object")
        operator = str(payload.get("operator", "")).lower()
        backend = str(payload.get("backend", "")).strip()
        version = str(payload.get("version", "")).strip()
        if not operator or not backend or not version:
            raise ValueError("kernel requires operator, backend and version")
        architectures = tuple(str(item) for item in (payload.get("architectures") or ["*"]))
        accelerators = tuple(str(item) for item in (payload.get("accelerators") or []))
        dtypes = tuple(str(item).lower() for item in (payload.get("dtypes") or ["fp32"]))
        if not architectures or not dtypes:
            raise ValueError("kernel architectures and dtypes cannot be empty")
        kernel_id = str(payload.get("id") or "")
        if not kernel_id:
            identity = json.dumps(
                {
                    "operator": operator,
                    "backend": backend,
                    "version": version,
                    "architectures": architectures,
                    "accelerators": accelerators,
                    "dtypes": dtypes,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            kernel_id = f"kernel-{hashlib.sha256(identity).hexdigest()[:16]}"
        constraints = payload.get("shape_constraints") or {}
        metadata = payload.get("metadata") or {}
        if not isinstance(constraints, dict) or not isinstance(metadata, dict):
            raise ValueError("kernel shape_constraints and metadata must be objects")
        return cls(kernel_id, operator, backend, version, architectures, accelerators, dtypes, constraints, metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "operator": self.operator,
            "backend": self.backend,
            "version": self.version,
            "architectures": list(self.architectures),
            "accelerators": list(self.accelerators),
            "dtypes": list(self.dtypes),
            "shape_constraints": self.shape_constraints,
            "metadata": self.metadata,
        }


def kernel_supports_worker(kernel: dict[str, Any], worker: dict[str, Any]) -> bool:
    capabilities = worker.get("capabilities") or {}
    architecture = capabilities.get("architecture")
    supported = set(kernel.get("architectures") or [])
    if "*" not in supported and architecture not in supported:
        return False
    required_accelerators = set(kernel.get("accelerators") or [])
    available_accelerators = set(capabilities.get("accelerators") or [])
    return required_accelerators.issubset(available_accelerators)


def kernel_supports_operator(kernel: dict[str, Any], operator: dict[str, Any]) -> bool:
    if kernel.get("operator") not in {operator.get("name"), "*"}:
        return False
    return operator.get("dtype", "fp32") in set(kernel.get("dtypes") or [])
