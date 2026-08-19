#!/usr/bin/env python3
"""Dependency-free, deterministic model-pipeline reference adapter.

This is a correctness and integration baseline, not an optimized compiler.
All files stay below the worker work root and are keyed by the EdgeForge
manifest digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _root() -> Path:
    digest = os.environ.get("EDGEFORGE_MODEL_MANIFEST_DIGEST", "standalone")
    return Path(".edgeforge") / "reference" / digest


def _infer(model: dict[str, Any], samples: list[list[float]]) -> list[list[float]]:
    outputs = []
    for sample in samples:
        logits = [sum(left * right for left, right in zip(sample, row)) + bias for row, bias in zip(model["weights"], model["bias"])]
        maximum = max(logits)
        exponentials = [math.exp(value - maximum) for value in logits]
        denominator = sum(exponentials)
        outputs.append([value / denominator for value in exponentials])
    return outputs


def export(root: Path) -> dict[str, Any]:
    model = {
        "format": "edgeforge-reference-linear-v1",
        "input_features": 16,
        "classes": 4,
        "weights": [[math.sin((row + 1) * (column + 1)) * 0.1 for column in range(16)] for row in range(4)],
        "bias": [0.01, -0.02, 0.03, -0.04],
    }
    _write(root / "exported-model.json", model)
    return {"format": model["format"], "model_digest": _digest(model), "path": str(root / "exported-model.json")}


def transform(root: Path) -> dict[str, Any]:
    raw = [[math.sin(sample * 0.17 + feature * 0.11) + 0.01 * sample for feature in range(16)] for sample in range(64)]
    normalized = []
    for sample in raw:
        mean = statistics.fmean(sample)
        variance = statistics.fmean([(value - mean) ** 2 for value in sample])
        scale = math.sqrt(variance) or 1.0
        normalized.append([(value - mean) / scale for value in sample])
    dataset = {"format": "edgeforge-synthetic-eeg-v1", "samples": normalized, "window_length": 16, "stride": 16}
    _write(root / "transformed-dataset.json", dataset)
    return {"dataset_digest": _digest(dataset), "samples": len(normalized), "features": 16, "path": str(root / "transformed-dataset.json")}


def compile_reference(root: Path) -> dict[str, Any]:
    model = _read(root / "exported-model.json")
    compiled = {"backend": "python-reference", "compiler_identity": "edgeforge-reference-v1", "model": model, "source_digest": _digest(model)}
    _write(root / "compiled-model.json", compiled)
    return {"backend": compiled["backend"], "compiler_identity": compiled["compiler_identity"], "compiled_digest": _digest(compiled), "path": str(root / "compiled-model.json")}


def run(root: Path) -> dict[str, Any]:
    compiled = _read(root / "compiled-model.json")
    dataset = _read(root / "transformed-dataset.json")
    outputs = _infer(compiled["model"], dataset["samples"])
    document = {"outputs": outputs}
    _write(root / "outputs.json", document)
    return {"outputs": len(outputs), "output_digest": _digest(document), "path": str(root / "outputs.json")}


def correctness(root: Path) -> dict[str, Any]:
    model = _read(root / "exported-model.json")
    samples = _read(root / "transformed-dataset.json")["samples"]
    actual = _read(root / "outputs.json")["outputs"]
    expected = _infer(model, samples)
    maximum_error = max(abs(left - right) for left_row, right_row in zip(actual, expected) for left, right in zip(left_row, right_row))
    return {"correctness": maximum_error <= 1e-12, "max_abs_error": maximum_error, "atol": 1e-12, "rtol": 0.0}


def benchmark(root: Path) -> dict[str, Any]:
    compiled = _read(root / "compiled-model.json")
    samples = _read(root / "transformed-dataset.json")["samples"]
    started = time.perf_counter()
    output = _infer(compiled["model"], samples)
    first_call_ms = (time.perf_counter() - started) * 1000.0
    timings = []
    for _ in range(20):
        started = time.perf_counter()
        output = _infer(compiled["model"], samples)
        timings.append((time.perf_counter() - started) * 1000.0)
    return {
        "first_call_ms": round(first_call_ms, 6),
        "steady_latency_ms": round(statistics.median(timings), 6),
        "summary": {"runs": len(timings), "min_ms": round(min(timings), 6), "median_ms": round(statistics.median(timings), 6), "p95_ms": round(sorted(timings)[18], 6)},
        "output_digest": _digest(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("export", "transform", "compile", "run", "correctness", "benchmark"))
    args = parser.parse_args()
    root = _root()
    handlers = {"export": export, "transform": transform, "compile": compile_reference, "run": run, "correctness": correctness, "benchmark": benchmark}
    print(json.dumps(handlers[args.stage](root), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
