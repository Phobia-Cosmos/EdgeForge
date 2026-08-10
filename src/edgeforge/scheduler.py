"""Hard-constraint filtering and deterministic worker scoring."""

from __future__ import annotations

from typing import Any


def _matches(worker: dict[str, Any], requirements: dict[str, Any]) -> bool:
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
    candidates = [worker for worker in workers if _matches(worker, requirements)]
    if not candidates:
        return None
    return max(candidates, key=lambda worker: (worker_score(worker, requirements), worker["id"]))

