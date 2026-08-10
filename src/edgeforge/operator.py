"""Small dependency-free Operator IR and reference benchmark backend."""

from __future__ import annotations

import hashlib
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable


SUPPORTED_OPERATORS = {"matmul", "softmax", "rmsnorm", "silu"}
SUPPORTED_DTYPES = {"fp32", "fp16", "bf16"}


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str = "fp32"
    backend: str = "python-reference"
    attrs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "OperatorSpec":
        if not isinstance(payload, dict):
            raise ValueError("operator must be an object")
        name = str(payload.get("name", "")).lower()
        if name not in SUPPORTED_OPERATORS:
            raise ValueError(f"unsupported operator: {name}")
        raw_shape = payload.get("shape")
        if not isinstance(raw_shape, list) or not raw_shape or not all(isinstance(item, int) for item in raw_shape):
            raise ValueError("operator.shape must be a non-empty list of integers")
        shape = tuple(raw_shape)
        if any(item <= 0 or item > 1024 for item in shape):
            raise ValueError("operator dimensions must be between 1 and 1024")
        expected_rank = 3 if name == "matmul" else 1
        if len(shape) != expected_rank:
            raise ValueError(f"{name} expects shape rank {expected_rank}")
        if name == "matmul" and shape[0] * shape[1] * shape[2] > 16_777_216:
            raise ValueError("matmul workload is too large for the reference backend")
        dtype = str(payload.get("dtype", "fp32")).lower()
        if dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported dtype: {dtype}")
        backend = str(payload.get("backend", "python-reference"))
        attrs = payload.get("attrs") or {}
        if not isinstance(attrs, dict):
            raise ValueError("operator.attrs must be an object")
        return cls(name=name, shape=shape, dtype=dtype, backend=backend, attrs=attrs)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "shape": list(self.shape), "dtype": self.dtype, "backend": self.backend, "attrs": self.attrs}


def _values(size: int) -> list[float]:
    return [((index * 17 + 3) % 29 - 14) / 7.0 for index in range(size)]


def _matmul(spec: OperatorSpec) -> tuple[list[float], Callable[[], bool]]:
    m, k, n = spec.shape
    left = _values(m * k)
    right = _values(k * n)
    output = [0.0] * (m * n)
    for row in range(m):
        for col in range(n):
            output[row * n + col] = sum(left[row * k + inner] * right[inner * n + col] for inner in range(k))

    def check() -> bool:
        samples = {(0, 0), (m - 1, n - 1), (m // 2, n // 2)}
        return all(
            math.isclose(
                output[row * n + col],
                sum(left[row * k + inner] * right[inner * n + col] for inner in range(k)),
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
            for row, col in samples
        )

    return output, check


def _softmax(spec: OperatorSpec) -> tuple[list[float], Callable[[], bool]]:
    values = _values(spec.shape[0])
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    denominator = sum(exponentials)
    output = [value / denominator for value in exponentials]
    return output, lambda: math.isclose(sum(output), 1.0, rel_tol=1e-6, abs_tol=1e-6) and all(value >= 0 for value in output)


def _rmsnorm(spec: OperatorSpec) -> tuple[list[float], Callable[[], bool]]:
    values = _values(spec.shape[0])
    epsilon = float(spec.attrs.get("epsilon", 1e-5))
    scale = math.sqrt(sum(value * value for value in values) / len(values) + epsilon)
    output = [value / scale for value in values]
    return output, lambda: all(math.isfinite(value) for value in output)


def _silu(spec: OperatorSpec) -> tuple[list[float], Callable[[], bool]]:
    values = _values(spec.shape[0])
    output = [value / (1.0 + math.exp(-value)) for value in values]
    return output, lambda: math.isclose(output[values.index(0.0)], 0.0, abs_tol=1e-8) if 0.0 in values else all(math.isfinite(value) for value in output)


def _run_once(spec: OperatorSpec) -> tuple[list[float], Callable[[], bool]]:
    return {"matmul": _matmul, "softmax": _softmax, "rmsnorm": _rmsnorm, "silu": _silu}[spec.name](spec)


def _checksum(values: list[float]) -> str:
    payload = ",".join(f"{value:.9g}" for value in values).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def benchmark_operator(spec: OperatorSpec, repeats: int = 3) -> dict[str, Any]:
    repeats = min(100, max(1, int(repeats)))
    timings: list[float] = []
    output: list[float] = []
    correctness = True
    errors: list[str] = []
    for _ in range(repeats):
        started = time.perf_counter()
        output, check = _run_once(spec)
        timings.append(round((time.perf_counter() - started) * 1000.0, 3))
        try:
            correctness = bool(check())
        except Exception as error:
            correctness = False
            errors.append(str(error))
        if not correctness:
            break
    ordered = sorted(timings)
    p95_index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.95) - 1))
    return {
        "operator": spec.to_dict(),
        "backend": spec.backend,
        "correctness": correctness,
        "errors": errors,
        "checksum": _checksum(output),
        "timings_ms": timings,
        "summary": {
            "runs": len(timings),
            "min_ms": min(timings),
            "median_ms": statistics.median(timings),
            "p95_ms": ordered[p95_index],
        },
    }
