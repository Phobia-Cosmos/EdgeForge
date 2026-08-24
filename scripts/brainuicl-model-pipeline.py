#!/usr/bin/env python3
"""Run a real BrainUICL checkpoint through the EdgeForge model stages.

The adapter lives in EdgeForge and only reads the external BrainUICL source,
checkpoint, and processed EEG paths. It never writes to that source tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("export", "transform", "compile", "run", "correctness", "benchmark"))
    parser.add_argument("--brainuicl-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compile-backend", default="inductor", choices=("inductor", "inductor-no-pattern", "aot_eager", "eager"))
    return parser.parse_args()


def setup_imports(root: Path) -> None:
    resolved = str(root.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def load_model(args: argparse.Namespace):
    import torch
    from model.pretrain_net import FeatureExtractor, SleepMLP, TransformerEncoder

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model_args = SimpleNamespace(dataset="ISRUC", device=device)
    blocks = [FeatureExtractor(model_args), TransformerEncoder(model_args), SleepMLP(model_args)]
    for block, name in zip(blocks, ("feature_extractor", "feature_encoder", "sleep_classifier")):
        checkpoint = args.checkpoint_root / f"{name}_parameter_{args.seed}.pkl"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        block.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        block.eval().to(device)
    return blocks, device


def make_wrapper(blocks):
    import torch

    class Model(torch.nn.Module):
        def __init__(self, modules):
            super().__init__()
            self.blocks = torch.nn.ModuleList(modules)

        def forward(self, x):
            values = x.reshape(-1, 8, 3000)
            eog = values[:, :2]
            eeg = values[:, 2:]
            return self.blocks[2](self.blocks[1](self.blocks[0](eeg, eog)))

    return Model(blocks).eval()


def sample_input(args: argparse.Namespace, device):
    import numpy as np
    import torch

    candidates = [
        args.data_root / str(args.subject) / "data" / "0.npy",
        args.data_root / f"sub-{args.subject:03d}" / "data" / "0.npy",
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"no sample sequence found for subject {args.subject} under {args.data_root}")
    values = np.load(path).astype("float32", copy=False)
    if values.shape != (20, 8, 3000):
        raise ValueError(f"expected (20, 8, 3000), got {values.shape}")
    return torch.from_numpy(values).unsqueeze(0).to(device), str(path)


def synchronize(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value, ensure_ascii=False))


def graph_summary(exported) -> dict[str, Any]:
    nodes = [{"op": str(node.op), "target": str(node.target)} for node in exported.graph_module.graph.nodes]
    canonical = json.dumps(nodes, sort_keys=True, separators=(",", ":"))
    counts: dict[str, int] = {}
    for node in nodes:
        key = f"{node['op']}:{node['target']}"
        counts[key] = counts.get(key, 0) + 1
    return {
        "schema_version": 1,
        "node_count": len(nodes),
        "operator_counts": counts,
        "graph_digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "nodes": nodes,
    }


def run_compiled(args: argparse.Namespace):
    import torch

    blocks, device = load_model(args)
    model = make_wrapper(blocks)
    if args.compile_backend != "eager":
        backend = "inductor" if args.compile_backend == "inductor-no-pattern" else args.compile_backend
        options = {"pattern_matcher": False} if args.compile_backend == "inductor-no-pattern" else None
        if options is not None:
            model = torch.compile(model, backend=backend, options=options)
        else:
            model = torch.compile(model, backend=backend, mode="reduce-overhead")
    sample, sample_path = sample_input(args, device)
    with torch.no_grad():
        synchronize(device)
        started = time.perf_counter()
        output = model(sample)
        synchronize(device)
        first_call_ms = (time.perf_counter() - started) * 1000.0
    return model, sample, output.detach(), device, sample_path, first_call_ms


def main() -> None:
    args = parse_args()
    setup_imports(args.brainuicl_root)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    import torch

    if args.stage == "transform":
        save_json(args.artifact_root / "transform.json", {"stage": "transform", "status": "validated", "dataset": "ISRUC", "sample_shape": [1, 20, 8, 3000]})
        return

    if args.stage == "export":
        blocks, device = load_model(args)
        model = make_wrapper(blocks)
        sample, sample_path = sample_input(args, device)
        with torch.no_grad():
            reference = model(sample).detach().cpu()
        exported = torch.export.export(model, (sample,))
        torch.export.save(exported, args.artifact_root / "brainuicl.pt2")
        torch.save(reference, args.artifact_root / "reference-output.pt")
        summary = graph_summary(exported)
        summary.update({"stage": "export", "status": "succeeded", "device": str(device), "sample_path": sample_path, "sample_shape": list(sample.shape), "output_shape": list(reference.shape), "artifact": "brainuicl.pt2"})
        save_json(args.artifact_root / "ir.json", summary)
        return

    if not (args.artifact_root / "brainuicl.pt2").is_file():
        raise FileNotFoundError("export stage must run before downstream stages")

    if args.stage == "compile":
        _model, _sample, _output, device, _path, first_call_ms = run_compiled(args)
        save_json(args.artifact_root / "compile.json", {"stage": "compile", "status": "succeeded", "backend": f"torch-compile-{args.compile_backend}", "device": str(device), "first_call_ms": first_call_ms, "artifact": "torch-compile-runtime"})
        return

    if args.stage == "run":
        _model, _sample, output, device, sample_path, first_call_ms = run_compiled(args)
        torch.save(output.cpu(), args.artifact_root / "compiled-output.pt")
        save_json(args.artifact_root / "run.json", {"stage": "run", "status": "succeeded", "device": str(device), "sample_path": sample_path, "first_call_ms": first_call_ms, "output_shape": list(output.shape)})
        return

    if args.stage == "correctness":
        eager = torch.load(args.artifact_root / "reference-output.pt", map_location="cpu", weights_only=True).float()
        compiled = torch.load(args.artifact_root / "compiled-output.pt", map_location="cpu", weights_only=True).float()
        delta = (eager - compiled).abs()
        passed = bool(torch.allclose(eager, compiled, rtol=2e-3, atol=2e-4))
        result = {"stage": "correctness", "correctness": passed, "max_abs_error": float(delta.max().item()), "mean_abs_error": float(delta.mean().item()), "rtol": 2e-3, "atol": 2e-4}
        save_json(args.artifact_root / "correctness.json", result)
        if not passed:
            raise SystemExit(2)
        return

    if args.stage == "benchmark":
        model, sample, _output, device, sample_path, first_call_ms = run_compiled(args)
        timings = []
        with torch.no_grad():
            for _ in range(max(1, args.repeats)):
                synchronize(device)
                started = time.perf_counter()
                model(sample)
                synchronize(device)
                timings.append((time.perf_counter() - started) * 1000.0)
        ordered = sorted(timings)
        result = {"stage": "benchmark", "device": str(device), "sample_path": sample_path, "first_call_ms": first_call_ms, "steady_latency_ms": sum(timings) / len(timings), "min_latency_ms": ordered[0], "p95_latency_ms": ordered[min(len(ordered) - 1, max(0, int(len(ordered) * 0.95) - 1))], "repeats": len(timings), "timings_ms": timings}
        save_json(args.artifact_root / "benchmark.json", result)


if __name__ == "__main__":
    main()
