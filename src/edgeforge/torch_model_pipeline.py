"""Optional PyTorch model-pipeline adapter for the local research Worker.

The control plane does not import this module.  A Worker runs it in an
environment that explicitly provides PyTorch, which keeps framework versions
and CUDA availability out of EdgeForge's core dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn


FEATURES = 16
HIDDEN = 32
CLASSES = 4


def _root() -> Path:
    digest = os.environ.get("EDGEFORGE_MODEL_MANIFEST_DIGEST", "standalone")
    return Path(".edgeforge") / "torch" / digest


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _tensor_digest(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous().view(-1).tolist()
    return _digest({"shape": list(tensor.shape), "dtype": str(tensor.dtype), "values": values})


def _model() -> nn.Module:
    torch.manual_seed(20260821)
    model = nn.Sequential(nn.Linear(FEATURES, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, CLASSES))
    model.eval()
    return model


def _load_model(root: Path) -> nn.Module:
    model = _model()
    state = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _load_dataset(root: Path) -> torch.Tensor:
    dataset = torch.load(root / "dataset.pt", map_location="cpu", weights_only=True)
    if not isinstance(dataset, torch.Tensor):
        raise RuntimeError("dataset.pt must contain a Tensor")
    return dataset.float()


def _compile(model: nn.Module) -> tuple[nn.Module, dict[str, Any]]:
    backend = os.environ.get("EDGEFORGE_MODEL_BACKEND", "torch-eager")
    if backend == "torch-eager":
        return model, {"backend": backend, "implementation": "torch.nn.Module", "compile_backend": None}
    if backend != "torch-compile":
        raise RuntimeError(f"unsupported PyTorch adapter backend: {backend}")
    compile_backend = os.environ.get("EDGEFORGE_TORCH_COMPILE_BACKEND", "eager")
    compiled = torch.compile(model, backend=compile_backend, fullgraph=True)
    return compiled, {
        "backend": backend,
        "implementation": "torch.compile",
        "compile_backend": compile_backend,
    }


def _forward(model: nn.Module, dataset: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        return model(dataset).detach().cpu()


def export(root: Path) -> dict[str, Any]:
    model = _model()
    root.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), root / "model.pt")
    metadata = {"format": "edgeforge-torch-mlp-v1", "features": FEATURES, "hidden": HIDDEN, "classes": CLASSES}
    _write_json(root / "model.json", metadata)
    return {"format": metadata["format"], "model_digest": _digest(metadata), "path": str(root / "model.pt")}


def transform(root: Path) -> dict[str, Any]:
    sample = torch.arange(64 * FEATURES, dtype=torch.float32).reshape(64, FEATURES)
    sample = torch.sin(sample * 0.017) + sample * 0.001
    mean = sample.mean(dim=1, keepdim=True)
    scale = sample.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    dataset = (sample - mean) / scale
    torch.save(dataset, root / "dataset.pt")
    return {"dataset_digest": _tensor_digest(dataset), "samples": dataset.shape[0], "features": dataset.shape[1], "path": str(root / "dataset.pt")}


def compile_model(root: Path) -> dict[str, Any]:
    model = _load_model(root)
    dataset = _load_dataset(root)
    started = time.perf_counter()
    compiled, metadata = _compile(model)
    # Force graph capture for torch.compile so compile_ms measures a real first compile.
    _forward(compiled, dataset[:1])
    compile_ms = (time.perf_counter() - started) * 1000.0
    metadata.update({"compile_ms": round(compile_ms, 6), "torch_version": torch.__version__})
    _write_json(root / "compiled.json", metadata)
    return {**metadata, "path": str(root / "compiled.json")}


def run(root: Path) -> dict[str, Any]:
    model = _load_model(root)
    dataset = _load_dataset(root)
    compiled, metadata = _compile(model)
    output = _forward(compiled, dataset)
    document = {"outputs": output.tolist(), "dtype": str(output.dtype), "shape": list(output.shape)}
    _write_json(root / "outputs.json", document)
    return {"outputs": len(output), "output_digest": _tensor_digest(output), "backend": metadata["backend"], "path": str(root / "outputs.json")}


def correctness(root: Path) -> dict[str, Any]:
    model = _load_model(root)
    dataset = _load_dataset(root)
    actual = torch.tensor(_read_json(root / "outputs.json")["outputs"], dtype=torch.float32)
    expected = _forward(model, dataset)
    difference = (actual - expected).abs()
    maximum_error = float(difference.max().item())
    passed = bool(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))
    return {"correctness": passed, "max_abs_error": maximum_error, "atol": 1e-5, "rtol": 1e-5}


def benchmark(root: Path) -> dict[str, Any]:
    model = _load_model(root)
    dataset = _load_dataset(root)
    compiled, metadata = _compile(model)
    started = time.perf_counter()
    output = _forward(compiled, dataset)
    first_call_ms = (time.perf_counter() - started) * 1000.0
    repeats = min(100, max(1, int(os.environ.get("EDGEFORGE_BENCHMARK_REPEATS", "10"))))
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        output = _forward(compiled, dataset)
        timings.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(timings)
    p95_index = min(len(ordered) - 1, max(0, int(len(ordered) * 0.95) - 1))
    return {
        "backend": metadata["backend"],
        "compile_backend": metadata["compile_backend"],
        "torch_version": torch.__version__,
        "first_call_ms": round(first_call_ms, 6),
        "steady_latency_ms": round(statistics.median(timings), 6),
        "summary": {"runs": len(timings), "min_ms": round(min(timings), 6), "median_ms": round(statistics.median(timings), 6), "p95_ms": round(ordered[p95_index], 6)},
        "output_digest": _tensor_digest(output),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("export", "transform", "compile", "run", "correctness", "benchmark"))
    args = parser.parse_args(argv)
    handlers = {"export": export, "transform": transform, "compile": compile_model, "run": run, "correctness": correctness, "benchmark": benchmark}
    print(json.dumps(handlers[args.stage](_root()), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
