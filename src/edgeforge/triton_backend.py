"""Optional Triton MatMul backend, imported only by GPU workers."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl

from edgeforge import __version__
from edgeforge.autotune import normalize_triton_matmul_candidates, normalize_triton_matmul_config, select_best_candidate
from edgeforge.operator import OperatorSpec


@triton.jit
def _matmul_kernel(
    left,
    right,
    output,
    m_size: tl.constexpr,
    n_size: tl.constexpr,
    k_size: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    program = tl.program_id(axis=0)
    programs_n = tl.cdiv(n_size, block_n)
    program_m = program // programs_n
    program_n = program % programs_n
    offsets_m = program_m * block_m + tl.arange(0, block_m)
    offsets_n = program_n * block_n + tl.arange(0, block_n)
    offsets_k = tl.arange(0, block_k)
    left_ptrs = left + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
    right_ptrs = right + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for start in range(0, tl.cdiv(k_size, block_k)):
        left_values = tl.load(left_ptrs, mask=(offsets_m[:, None] < m_size) & (offsets_k[None, :] + start * block_k < k_size), other=0.0)
        right_values = tl.load(right_ptrs, mask=(offsets_k[:, None] + start * block_k < k_size) & (offsets_n[None, :] < n_size), other=0.0)
        accumulator += tl.dot(left_values, right_values)
        left_ptrs += block_k * stride_ak
        right_ptrs += block_k * stride_bk
    output_values = accumulator.to(tl.float16)
    output_ptrs = output + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    tl.store(output_ptrs, output_values, mask=(offsets_m[:, None] < m_size) & (offsets_n[None, :] < n_size))


def _launch(left: torch.Tensor, right: torch.Tensor, output: torch.Tensor, config: dict[str, int] | None = None) -> None:
    m_size, k_size = left.shape
    _, n_size = right.shape
    selected = normalize_triton_matmul_config(config or {})
    block_m = selected["block_m"]
    block_n = selected["block_n"]
    block_k = selected["block_k"]
    grid = (triton.cdiv(m_size, block_m) * triton.cdiv(n_size, block_n),)
    _matmul_kernel[grid](
        left,
        right,
        output,
        m_size,
        n_size,
        k_size,
        left.stride(0),
        left.stride(1),
        right.stride(0),
        right.stride(1),
        output.stride(0),
        output.stride(1),
        block_m,
        block_n,
        block_k,
        num_warps=selected["num_warps"],
        num_stages=selected["num_stages"],
    )


def run_triton_matmul_pipeline(spec: OperatorSpec, kernel: dict[str, Any], repeats: int, warmup: int) -> dict[str, Any]:
    if spec.name != "matmul":
        raise ValueError("the Triton backend currently supports only matmul")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to this worker")
    if spec.dtype not in {"fp16", "bf16"}:
        raise ValueError("the Triton MatMul backend currently supports fp16 and bf16")
    m_size, k_size, n_size = spec.shape
    torch_dtype = torch.float16 if spec.dtype == "fp16" else torch.bfloat16
    tuning_config = normalize_triton_matmul_config((kernel.get("metadata") or {}).get("tuning_config") or {})
    torch.manual_seed(0)
    left = torch.randn((m_size, k_size), device="cuda", dtype=torch_dtype)
    right = torch.randn((k_size, n_size), device="cuda", dtype=torch_dtype)
    output = torch.empty((m_size, n_size), device="cuda", dtype=torch_dtype)
    stages = []

    started = time.perf_counter()
    _launch(left, right, output, tuning_config)
    torch.cuda.synchronize()
    compile_ms = (time.perf_counter() - started) * 1000.0
    stages.append({"name": "compile", "status": "succeeded", "elapsed_ms": round(compile_ms, 3), "details": {"first_launch_includes_jit": True}})

    started = time.perf_counter()
    reference = torch.matmul(left, right)
    max_abs_error = float((output - reference).abs().max().item())
    tolerance = 0.08 if spec.dtype == "fp16" else 0.15
    correctness = bool(torch.allclose(output, reference, rtol=tolerance, atol=tolerance))
    correctness_ms = (time.perf_counter() - started) * 1000.0
    stages.append(
        {
            "name": "correctness",
            "status": "succeeded" if correctness else "failed",
            "elapsed_ms": round(correctness_ms, 3),
            "details": {"max_abs_error": max_abs_error, "rtol": tolerance, "atol": tolerance},
        }
    )

    timings = []
    if correctness:
        for _ in range(max(0, warmup)):
            _launch(left, right, output, tuning_config)
        torch.cuda.synchronize()
        for _ in range(repeats):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            _launch(left, right, output, tuning_config)
            end_event.record()
            end_event.synchronize()
            timings.append(round(float(start_event.elapsed_time(end_event)), 6))
        ordered = sorted(timings)
        p95_index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.95) - 1))
        summary = {
            "runs": len(timings),
            "min_ms": min(timings),
            "median_ms": float(torch.tensor(timings).median().item()) if len(timings) % 2 else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2,
            "p95_ms": ordered[p95_index],
        }
        stages.append({"name": "benchmark", "status": "succeeded", "elapsed_ms": round(sum(timings), 3), "details": {"summary": summary, "warmup": warmup}})
    else:
        summary = {"runs": 0}

    source_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "backend": "triton",
        "runtime_version": __version__,
        "operator": spec.to_dict(),
        "kernel": kernel,
        "compiler": {
            "torch": torch.__version__,
            "triton": triton.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "source_digest": source_digest,
            "tuning_config": tuning_config,
        },
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "operator": spec.to_dict(),
        "backend": "triton",
        "kernel_id": kernel.get("id"),
        "kernel_version": kernel.get("version"),
        "correctness": correctness,
        "max_abs_error": max_abs_error,
        "timings_ms": timings,
        "summary": summary,
        "pipeline": stages,
        "compile_ms": round(compile_ms, 3),
        "elapsed_ms": round(sum(stage["elapsed_ms"] for stage in stages), 3),
        "exit_code": 0 if correctness else 1,
        "artifact_upload": {
            "name": f"{kernel.get('id', 'triton-matmul')}-manifest.json",
            "kind": "compiler-manifest",
            "media_type": "application/json",
            "metadata": {"backend": "triton", "source_digest": source_digest, "runtime_version": __version__},
            "content_base64": base64.b64encode(manifest_bytes).decode("ascii"),
        },
    }


def run_triton_matmul_autotune(
    spec: OperatorSpec,
    kernel: dict[str, Any],
    candidates: list[dict[str, Any]],
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    if spec.name != "matmul":
        raise ValueError("the Triton Auto Tuner currently supports only matmul")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to this worker")
    if spec.dtype not in {"fp16", "bf16"}:
        raise ValueError("the Triton Auto Tuner currently supports fp16 and bf16")
    search_space = normalize_triton_matmul_candidates(candidates)
    m_size, k_size, n_size = spec.shape
    torch_dtype = torch.float16 if spec.dtype == "fp16" else torch.bfloat16
    torch.manual_seed(0)
    left = torch.randn((m_size, k_size), device="cuda", dtype=torch_dtype)
    right = torch.randn((k_size, n_size), device="cuda", dtype=torch_dtype)
    output = torch.empty((m_size, n_size), device="cuda", dtype=torch_dtype)
    reference = torch.matmul(left, right)
    results: list[dict[str, Any]] = []
    for config in search_space:
        candidate: dict[str, Any] = {"config": config, "status": "failed", "correctness": False}
        try:
            started = time.perf_counter()
            _launch(left, right, output, config)
            torch.cuda.synchronize()
            compile_ms = (time.perf_counter() - started) * 1000.0
            candidate["compile_ms"] = round(compile_ms, 3)
            max_abs_error = float((output - reference).abs().max().item())
            tolerance = 0.08 if spec.dtype == "fp16" else 0.15
            correctness = bool(torch.allclose(output, reference, rtol=tolerance, atol=tolerance))
            candidate["correctness"] = correctness
            candidate["max_abs_error"] = max_abs_error
            candidate["status"] = "succeeded" if correctness else "failed"
            if not correctness:
                candidate["error"] = "correctness tolerance exceeded"
                results.append(candidate)
                continue
            for _ in range(max(0, warmup)):
                _launch(left, right, output, config)
            torch.cuda.synchronize()
            timings = []
            for _ in range(repeats):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _launch(left, right, output, config)
                end_event.record()
                end_event.synchronize()
                timings.append(round(float(start_event.elapsed_time(end_event)), 6))
            ordered = sorted(timings)
            p95_index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.95) - 1))
            median_ms = (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2 if len(ordered) % 2 == 0 else ordered[len(ordered) // 2]
            candidate["timings_ms"] = timings
            candidate["summary"] = {
                "runs": len(timings),
                "min_ms": min(timings),
                "median_ms": median_ms,
                "p95_ms": ordered[p95_index],
            }
        except Exception as error:
            candidate["status"] = "failed"
            candidate["error"] = str(error)[:500]
        results.append(candidate)
    best = select_best_candidate(results)
    source_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "backend": "triton",
        "runtime_version": __version__,
        "operator": spec.to_dict(),
        "kernel": kernel,
        "search_space": search_space,
        "candidates": results,
        "best_config": best.get("config") if best else None,
        "compiler": {
            "torch": torch.__version__,
            "triton": triton.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "source_digest": source_digest,
        },
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    correctness = best is not None
    best_summary = best.get("summary") if best else {"runs": 0}
    pipeline = [
        {"name": "compile", "status": "succeeded" if any(item.get("compile_ms") is not None for item in results) else "failed", "elapsed_ms": round(sum(float(item.get("compile_ms") or 0.0) for item in results), 3), "details": {"candidate_count": len(results)}},
        {"name": "correctness", "status": "succeeded" if correctness else "failed", "elapsed_ms": 0.0, "details": {"successful_candidates": sum(1 for item in results if item.get("correctness"))}},
        {"name": "benchmark", "status": "succeeded" if correctness else "failed", "elapsed_ms": round(sum(sum(float(value) for value in item.get("timings_ms", [])) for item in results), 3), "details": {"best_config": best.get("config") if best else None}},
    ]
    return {
        "operator": spec.to_dict(),
        "backend": "triton",
        "kernel_id": kernel.get("id"),
        "kernel_version": kernel.get("version"),
        "correctness": correctness,
        "timings_ms": (best or {}).get("timings_ms", []),
        "summary": best_summary,
        "compile_ms": (best or {}).get("compile_ms"),
        "best_config": (best or {}).get("config"),
        "search_space": search_space,
        "candidates": results,
        "pipeline": pipeline,
        "elapsed_ms": round(sum(stage["elapsed_ms"] for stage in pipeline), 3),
        "exit_code": 0 if correctness else 1,
        "artifact_upload": {
            "name": f"{kernel.get('id', 'triton-matmul')}-autotune-manifest.json",
            "kind": "autotune-manifest",
            "media_type": "application/json",
            "metadata": {"backend": "triton", "source_digest": source_digest, "runtime_version": __version__},
            "content_base64": base64.b64encode(manifest_bytes).decode("ascii"),
        },
    }
