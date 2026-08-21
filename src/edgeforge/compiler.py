"""Compiler backend dispatch and compile/correctness/benchmark pipelines."""

from __future__ import annotations

import base64
import json
import time
from typing import Any

from edgeforge import __version__
from edgeforge.operator import OperatorSpec, benchmark_operator


def _stage(name: str, status: str, elapsed_ms: float, details: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "status": status, "elapsed_ms": round(elapsed_ms, 3), "details": details}


def _artifact_upload(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    content = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "name": name,
        "kind": "compiler-manifest",
        "media_type": "application/json",
        "metadata": {"backend": manifest["backend"], "runtime_version": __version__},
        "content_base64": base64.b64encode(content).decode("ascii"),
    }


def _reference_pipeline(spec: OperatorSpec, kernel: dict[str, Any], repeats: int) -> dict[str, Any]:
    stages = []
    started = time.perf_counter()
    manifest = {
        "schema_version": 1,
        "backend": "python-reference",
        "runtime_version": __version__,
        "operator": spec.to_dict(),
        "kernel": kernel,
        "compiler": {"name": "python", "mode": "interpreted-reference"},
    }
    compile_ms = (time.perf_counter() - started) * 1000.0
    stages.append(_stage("compile", "succeeded", compile_ms, {"manifest_generated": True}))

    started = time.perf_counter()
    correctness_result = benchmark_operator(spec, 1)
    correctness_ms = (time.perf_counter() - started) * 1000.0
    correctness_status = "succeeded" if correctness_result["correctness"] else "failed"
    stages.append(
        _stage(
            "correctness",
            correctness_status,
            correctness_ms,
            {"checksum": correctness_result["checksum"], "errors": correctness_result["errors"]},
        )
    )
    if not correctness_result["correctness"]:
        return {
            "operator": spec.to_dict(),
            "backend": "python-reference",
            "kernel_id": kernel.get("id"),
            "kernel_version": kernel.get("version"),
            "correctness": False,
            "pipeline": stages,
            "compile_ms": round(compile_ms, 3),
            "exit_code": 1,
            "artifact_upload": _artifact_upload(manifest, f"{kernel.get('id', spec.name)}-manifest.json"),
        }

    started = time.perf_counter()
    benchmark = benchmark_operator(spec, repeats)
    benchmark_ms = (time.perf_counter() - started) * 1000.0
    stages.append(_stage("benchmark", "succeeded", benchmark_ms, {"summary": benchmark["summary"]}))
    return {
        **benchmark,
        "kernel_id": kernel.get("id"),
        "kernel_version": kernel.get("version"),
        "pipeline": stages,
        "compile_ms": round(compile_ms, 3),
        "elapsed_ms": round(sum(stage["elapsed_ms"] for stage in stages), 3),
        "exit_code": 0,
        "artifact_upload": _artifact_upload(manifest, f"{kernel.get('id', spec.name)}-manifest.json"),
    }


def run_kernel_pipeline(payload: dict[str, Any]) -> dict[str, Any]:
    spec = OperatorSpec.from_payload(payload.get("operator") or {})
    kernel = payload.get("kernel") or {}
    if not kernel:
        raise ValueError("kernel_pipeline requires a registered kernel snapshot")
    repeats = min(100, max(1, int(payload.get("repeats") or 5)))
    backend = kernel.get("backend") or spec.backend
    if backend == "python-reference":
        return _reference_pipeline(spec, kernel, repeats)
    if backend == "triton":
        from edgeforge.triton_backend import run_triton_matmul_pipeline

        return run_triton_matmul_pipeline(spec, kernel, repeats, int(payload.get("warmup") or 5))
    if backend == "iree-ukernel":
        from edgeforge.iree_backend import run_iree_conv_nchwc_pipeline

        return run_iree_conv_nchwc_pipeline(
            spec,
            kernel,
            repeats,
            work_root=payload.get("_worker_work_root"),
            allowed_commands=payload.get("_allowed_commands"),
        )
    raise ValueError(f"unsupported compiler backend: {backend}")


def run_kernel_autotune(payload: dict[str, Any]) -> dict[str, Any]:
    spec = OperatorSpec.from_payload(payload.get("operator") or {})
    kernel = payload.get("kernel") or {}
    if not kernel:
        raise ValueError("kernel_autotune requires a registered kernel snapshot")
    backend = kernel.get("backend") or spec.backend
    if backend != "triton":
        raise ValueError("the first Auto Tuner supports only Triton kernels")
    from edgeforge.triton_backend import run_triton_matmul_autotune

    return run_triton_matmul_autotune(
        spec,
        kernel,
        payload.get("candidates") or [],
        min(100, max(1, int(payload.get("repeats") or 5))),
        min(100, max(0, int(payload.get("warmup") or 3))),
    )
