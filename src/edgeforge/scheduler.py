"""Hard constraints, worker scoring and compiler-aware execution planning."""

from __future__ import annotations

import statistics
from typing import Any

from edgeforge.kernel import kernel_supports_operator, kernel_supports_worker


def worker_matches(worker: dict[str, Any], requirements: dict[str, Any]) -> bool:
    capabilities = worker.get("capabilities") or {}
    metrics = worker.get("metrics") or {}

    worker_ids = requirements.get("worker_ids") or []
    if worker_ids and worker["id"] not in worker_ids:
        return False

    architectures = requirements.get("architectures") or []
    if architectures and capabilities.get("architecture") not in architectures:
        return False

    accelerators = set(capabilities.get("accelerators") or [])
    required_accelerators = set(requirements.get("accelerators") or [])
    if not required_accelerators.issubset(accelerators):
        return False

    # `backends` is historically a Kernel-plan filter.  Model tasks use the
    # explicit Worker capability key so existing compiler-plan semantics stay
    # backwards compatible.
    required_backends = set(requirements.get("worker_backends") or [])
    if required_backends:
        advertised_backends = set(capabilities.get("backends") or [])
        # Older Workers predate backend advertisement; infer only the
        # dependency-free reference executor for backwards compatibility.
        if "python-reference" in required_backends and not advertised_backends:
            advertised_backends.add("python-reference")
        if not required_backends.issubset(advertised_backends):
            return False

    required_labels = requirements.get("labels") or {}
    labels = worker.get("labels") or {}
    if any(labels.get(key) != value for key, value in required_labels.items()):
        return False

    min_memory_mb = int(requirements.get("min_memory_mb") or 0)
    if min_memory_mb and int(metrics.get("memory_available_mb") or 0) < min_memory_mb:
        return False

    return True


def worker_score(worker: dict[str, Any], requirements: dict[str, Any]) -> float:
    """Return a higher-is-better score after hard constraints have matched."""
    capabilities = worker.get("capabilities") or {}
    metrics = worker.get("metrics") or {}
    cpu_count = max(1, int(capabilities.get("cpu_count") or 1))
    load_ratio = float(metrics.get("load_1m") or 0.0) / cpu_count
    memory_total = max(1, int(capabilities.get("memory_total_mb") or 1))
    memory_ratio = min(1.0, float(metrics.get("memory_available_mb") or 0) / memory_total)
    active_tasks = int(worker.get("active_tasks") or 0)

    score = 20.0 * memory_ratio - 30.0 * load_ratio - 15.0 * active_tasks
    preferred = set(requirements.get("prefer_accelerators") or [])
    available = set(capabilities.get("accelerators") or [])
    score += 25.0 * len(preferred.intersection(available))
    return score


def select_worker(
    workers: list[dict[str, Any]], requirements: dict[str, Any]
) -> dict[str, Any] | None:
    candidates = [worker for worker in workers if worker_matches(worker, requirements)]
    if not candidates:
        return None
    return max(candidates, key=lambda worker: (worker_score(worker, requirements), worker["id"]))


def _history_estimate(
    benchmarks: list[dict[str, Any]],
    worker: dict[str, Any],
    kernel: dict[str, Any],
    operator: dict[str, Any],
) -> tuple[float | None, float, str, int]:
    capabilities = worker.get("capabilities") or {}
    architecture = capabilities.get("architecture")
    matching = [
        item
        for item in benchmarks
        if item.get("correctness")
        and item.get("operator") == operator.get("name")
        and item.get("shape") == operator.get("shape")
        and item.get("dtype") == operator.get("dtype", "fp32")
        and item.get("kernel_id") == kernel.get("id")
    ]
    exact = [item for item in matching if item.get("worker_id") == worker.get("id")]
    samples = exact or [item for item in matching if item.get("architecture") == architecture]
    if not samples:
        return None, 0.0, "unseen", 0
    latencies = [
        float(item.get("summary", {}).get("median_ms"))
        for item in samples
        if isinstance(item.get("summary", {}).get("median_ms"), (int, float))
        and float(item["summary"]["median_ms"]) > 0
    ]
    if not latencies:
        return None, 0.0, "unseen", 0
    compile_times = [
        float(item["compile_ms"])
        for item in samples
        if isinstance(item.get("compile_ms"), (int, float)) and float(item["compile_ms"]) >= 0
    ]
    return (
        float(statistics.median(latencies)),
        float(statistics.median(compile_times)) if compile_times else 0.0,
        "exact-worker" if exact else "same-architecture",
        len(latencies),
    )


def build_execution_plan(
    workers: list[dict[str, Any]],
    kernels: list[dict[str, Any]],
    benchmarks: list[dict[str, Any]],
    operator: dict[str, Any],
    requirements: dict[str, Any] | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    requirements = requirements or {}
    requested_policy = policy or {}
    compile_weight = max(0.0, float(requested_policy.get("compile_weight", 0.05)))
    load_weight_ms = max(0.0, float(requested_policy.get("load_weight_ms", 1.0)))
    unknown_latency_ms = max(0.001, float(requested_policy.get("unknown_latency_ms", 1000.0)))
    normalized_policy = {
        "compile_weight": compile_weight,
        "load_weight_ms": load_weight_ms,
        "unknown_latency_ms": unknown_latency_ms,
    }
    allowed_kernel_ids = set(requirements.get("kernel_ids") or [])
    allowed_backends = set(requirements.get("backends") or [])
    candidates = []
    for kernel in kernels:
        if kernel.get("status") != "active":
            continue
        if allowed_kernel_ids and kernel.get("id") not in allowed_kernel_ids:
            continue
        if allowed_backends and kernel.get("backend") not in allowed_backends:
            continue
        if not kernel_supports_operator(kernel, operator):
            continue
        for worker in workers:
            if worker.get("status") != "online":
                continue
            if not worker_matches(worker, requirements) or not kernel_supports_worker(kernel, worker):
                continue
            latency, compile_ms, source, sample_count = _history_estimate(benchmarks, worker, kernel, operator)
            capabilities = worker.get("capabilities") or {}
            metrics = worker.get("metrics") or {}
            cpu_count = max(1, int(capabilities.get("cpu_count") or 1))
            load_ratio = max(0.0, float(metrics.get("load_1m") or 0.0) / cpu_count)
            active_tasks = max(0, int(worker.get("active_tasks") or 0))
            load_penalty_ms = load_weight_ms * (load_ratio + active_tasks)
            estimated_latency_ms = latency if latency is not None else unknown_latency_ms
            objective_ms = estimated_latency_ms + compile_weight * compile_ms + load_penalty_ms
            candidates.append(
                {
                    "worker_id": worker["id"],
                    "kernel_id": kernel["id"],
                    "architecture": capabilities.get("architecture", "unknown"),
                    "backend": kernel.get("backend", "unknown"),
                    "estimate_source": source,
                    "sample_count": sample_count,
                    "estimated_latency_ms": round(estimated_latency_ms, 6),
                    "estimated_compile_ms": round(compile_ms, 6),
                    "load_penalty_ms": round(load_penalty_ms, 6),
                    "objective_ms": round(objective_ms, 6),
                }
            )
    candidates.sort(key=lambda item: (item["objective_ms"], item["worker_id"], item["kernel_id"]))
    if not candidates:
        raise ValueError("no compatible online Worker and Kernel execution path")
    selected = candidates[0]
    return {
        "operator": operator,
        "policy": normalized_policy,
        "selected": selected,
        "candidates": candidates,
        "reason": (
            f"selected {selected['worker_id']} with {selected['kernel_id']} using "
            f"{selected['estimate_source']} cost estimate ({selected['objective_ms']} ms objective)"
        ),
    }
