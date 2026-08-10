"""Backend-independent helpers for deterministic kernel parameter search."""

from __future__ import annotations

from typing import Any


_BLOCK_SIZES = {16, 32, 64, 128}
_NUM_WARPS = {1, 2, 4, 8}


def normalize_triton_matmul_config(value: dict[str, Any]) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError("each tuning candidate must be an object")
    config = {
        "block_m": int(value.get("block_m", 32)),
        "block_n": int(value.get("block_n", 32)),
        "block_k": int(value.get("block_k", 32)),
        "num_warps": int(value.get("num_warps", 4)),
        "num_stages": int(value.get("num_stages", 2)),
    }
    if config["block_m"] not in _BLOCK_SIZES or config["block_n"] not in _BLOCK_SIZES:
        raise ValueError("block_m and block_n must be one of 16, 32, 64 or 128")
    if config["block_k"] not in _BLOCK_SIZES:
        raise ValueError("block_k must be one of 16, 32, 64 or 128")
    if config["num_warps"] not in _NUM_WARPS:
        raise ValueError("num_warps must be one of 1, 2, 4 or 8")
    if not 1 <= config["num_stages"] <= 5:
        raise ValueError("num_stages must be between 1 and 5")
    return config


def default_triton_matmul_candidates() -> list[dict[str, int]]:
    return [
        {"block_m": 32, "block_n": 32, "block_k": 32, "num_warps": 4, "num_stages": 2},
        {"block_m": 64, "block_n": 32, "block_k": 32, "num_warps": 4, "num_stages": 2},
        {"block_m": 32, "block_n": 64, "block_k": 32, "num_warps": 4, "num_stages": 2},
        {"block_m": 64, "block_n": 64, "block_k": 32, "num_warps": 4, "num_stages": 3},
    ]


def normalize_triton_matmul_candidates(values: list[dict[str, Any]] | None) -> list[dict[str, int]]:
    source = values or default_triton_matmul_candidates()
    if len(source) > 64:
        raise ValueError("at most 64 tuning candidates are allowed")
    result = []
    seen = set()
    for value in source:
        config = normalize_triton_matmul_config(value)
        key = tuple(config.items())
        if key not in seen:
            seen.add(key)
            result.append(config)
    if not result:
        raise ValueError("the tuning search space must not be empty")
    return result


def select_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = []
    for index, candidate in enumerate(candidates):
        summary = candidate.get("summary") or {}
        median_ms = summary.get("median_ms")
        if candidate.get("status") != "succeeded" or not candidate.get("correctness"):
            continue
        if not isinstance(median_ms, (int, float)) or median_ms <= 0:
            continue
        p95_ms = summary.get("p95_ms", median_ms)
        compile_ms = candidate.get("compile_ms", float("inf"))
        valid.append((float(median_ms), float(p95_ms), float(compile_ms), index, candidate))
    return min(valid, key=lambda item: item[:4])[-1] if valid else None
