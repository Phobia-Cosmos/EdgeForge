#!/usr/bin/env python3
"""Compare registered EEG architectures on a tiny local ISRUC split.

This is a CPU-friendly pipeline benchmark, not a publication-grade result.
It intentionally writes all outputs to a caller-selected path outside Git.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from edgeforge.eeg_models import EEGModelConfig, build_eeg_decoder


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def macro_f1(predictions: torch.Tensor, labels: torch.Tensor, classes: int) -> float:
    values = []
    for cls in range(classes):
        tp = ((predictions == cls) & (labels == cls)).sum().item()
        fp = ((predictions == cls) & (labels != cls)).sum().item()
        fn = ((predictions != cls) & (labels == cls)).sum().item()
        denom = 2 * tp + fp + fn
        values.append(0.0 if denom == 0 else (2.0 * tp / denom))
    return float(np.mean(values))


def read_group(root: Path, group: str) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    values: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    names: list[str] = []
    for data_path in sorted((root / group).glob("*/data/*.npy"), key=lambda p: (int(p.parent.parent.name), int(p.stem))):
        label_path = root / group / data_path.parent.parent.name / "label" / data_path.name
        data = np.load(data_path, allow_pickle=False)
        target = np.load(label_path, allow_pickle=False).reshape(-1)
        if data.shape != (20, 8, 3000) or target.shape != (20,):
            raise ValueError(f"invalid pair {data_path}: {data.shape=} {target.shape=}")
        values.append(data.astype(np.float32, copy=False))
        labels.append(target.astype(np.int64, copy=False))
        names.append(str(data_path))
    if not values:
        raise RuntimeError(f"no files found for group {group} under {root}")
    return torch.from_numpy(np.concatenate(values, axis=0)), torch.from_numpy(np.concatenate(labels, axis=0)), names


def read_target_halves(root: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    train_values: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    eval_values: list[np.ndarray] = []
    eval_labels: list[np.ndarray] = []
    names: list[str] = []
    for data_path in sorted((root / "target").glob("*/data/*.npy"), key=lambda p: (int(p.parent.parent.name), int(p.stem))):
        label_path = root / "target" / data_path.parent.parent.name / "label" / data_path.name
        data = np.load(data_path, allow_pickle=False)
        target = np.load(label_path, allow_pickle=False).reshape(-1)
        if data.shape != (20, 8, 3000) or target.shape != (20,):
            raise ValueError(f"invalid pair {data_path}: {data.shape=} {target.shape=}")
        train_values.append(data[:10].astype(np.float32, copy=False))
        train_labels.append(target[:10].astype(np.int64, copy=False))
        eval_values.append(data[10:].astype(np.float32, copy=False))
        eval_labels.append(target[10:].astype(np.int64, copy=False))
        names.append(str(data_path))
    if not train_values:
        raise RuntimeError(f"no target files found under {root}")
    return (
        torch.from_numpy(np.concatenate(train_values, axis=0)),
        torch.from_numpy(np.concatenate(train_labels, axis=0)),
        torch.from_numpy(np.concatenate(eval_values, axis=0)),
        torch.from_numpy(np.concatenate(eval_labels, axis=0)),
        names,
    )


def model_config(name: str) -> EEGModelConfig:
    options = {}
    if name == "lop_mlp":
        options = {"hidden_dims": [64, 64]}
    if name == "brainuicl":
        options = {"d_model": 64, "classifier_hidden": 32}
    return EEGModelConfig(
        name=name,
        in_channels=8,
        input_length=3000,
        num_classes=5,
        feature_dim=32,
        width=16,
        depth=3,
        kernel_size=15,
        patch_size=64,
        layers=1,
        heads=4,
        dropout=0.0,
        options=options,
    )


def build(name: str) -> torch.nn.Module:
    return build_eeg_decoder(model_config(name))


@torch.no_grad()
def evaluate(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int, diagnostics: bool = False) -> dict:
    model.eval()
    losses: list[float] = []
    predictions: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    representations: list[torch.Tensor] = []
    attention_entropy: list[float] = []
    for batch_x, batch_y in DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False):
        output = model.forward_bundle(batch_x) if diagnostics else model(batch_x)
        logits = output.logits if diagnostics else output
        losses.append(float(F.cross_entropy(logits, batch_y).item()))
        predictions.append(logits.argmax(dim=-1).cpu())
        labels.append(batch_y.cpu())
        if diagnostics:
            preferred = output.representations.get("classifier_input", output.representations.get("embedding"))
            if preferred is not None:
                representations.append(preferred.detach().cpu().reshape(-1, preferred.shape[-1]))
            for weights in output.attention.values():
                probabilities = weights.float().clamp_min(1e-12)
                attention_entropy.append(float((-(probabilities * probabilities.log()).sum(-1).mean()).item()))
    pred = torch.cat(predictions)
    target = torch.cat(labels)
    result = {"loss": float(np.mean(losses)), "accuracy": float((pred == target).float().mean().item()), "macro_f1": macro_f1(pred, target, 5)}
    if diagnostics and representations:
        matrix = torch.cat(representations).float()
        matrix = matrix - matrix.mean(dim=0, keepdim=True)
        singular = torch.linalg.svdvals(matrix)
        energy = singular.square()
        probability = energy / energy.sum().clamp_min(1e-12)
        result["effective_rank"] = float(torch.exp(-(probability * probability.clamp_min(1e-12).log()).sum()).item())
        result["stable_rank"] = float((energy.sum() / singular.max().square().clamp_min(1e-12)).item())
    if attention_entropy:
        result["attention_entropy"] = float(np.mean(attention_entropy))
    return result


def train(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, epochs: int, batch_size: int, lr: float, seed: int) -> list[dict]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
    return history


def adapt(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, steps: int, batch_size: int, lr: float, seed: int) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    iterator = iter(loader)
    model.train()
    for _ in range(steps):
        try:
            batch_x, batch_y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch_x, batch_y = next(iterator)
        loss = F.cross_entropy(model(batch_x), batch_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def latency_ms(model: torch.nn.Module, x: torch.Tensor, repeats: int) -> float:
    model.eval()
    sample = x[: min(4, len(x))]
    with torch.no_grad():
        for _ in range(2):
            model(sample)
        started = time.perf_counter()
        for _ in range(repeats):
            model(sample)
    return (time.perf_counter() - started) * 1000.0 / repeats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--architectures", nargs="+", default=["lop_mlp", "eegnet", "tcn", "transformer", "brainuicl"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[4321, 4322, 4323])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--adapt-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--adapt-lr", type=float, default=1e-3)
    parser.add_argument("--latency-repeats", type=int, default=3)
    args = parser.parse_args()
    root = args.data_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_x, source_y, source_files = read_group(root, "source")
    target_train_x, target_train_y, target_eval_x, target_eval_y, target_files = read_target_halves(root)
    retention_x, retention_y, retention_files = read_group(root, "retention")
    metadata = {"schema": "edgeforge.eeg-architecture-benchmark.v1", "dataset": "ISRUC", "data_root": str(root), "source_files": source_files, "target_files": target_files, "retention_files": retention_files, "target_split": "first 10 epochs of every file for adaptation; last 10 for held-out evaluation", "input_shape": [8, 3000], "classes": 5, "epochs": args.epochs, "adapt_steps": args.adapt_steps, "device": "cpu", "scientific_conclusion_allowed": False}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    all_runs: list[dict] = []
    for name in args.architectures:
        for seed in args.seeds:
            seed_all(seed)
            started = time.time()
            model = build(name)
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            history = train(model, source_x, source_y, args.epochs, args.batch_size, args.lr, seed)
            source_metrics = evaluate(model, source_x, source_y, args.batch_size, diagnostics=True)
            checkpoint = copy.deepcopy(model)
            fresh = build(name)
            adapt(checkpoint, target_train_x, target_train_y, args.adapt_steps, args.batch_size, args.adapt_lr, seed + 11)
            adapt(fresh, target_train_x, target_train_y, args.adapt_steps, args.batch_size, args.adapt_lr, seed + 11)
            checkpoint_target = evaluate(checkpoint, target_eval_x, target_eval_y, args.batch_size, diagnostics=True)
            fresh_target = evaluate(fresh, target_eval_x, target_eval_y, args.batch_size, diagnostics=True)
            retention = evaluate(checkpoint, retention_x, retention_y, args.batch_size)
            result = {"architecture": name, "seed": seed, "parameters": parameter_count, "pretrain": history, "source": source_metrics, "target_checkpoint": checkpoint_target, "target_fresh": fresh_target, "retention": retention, "fresh_gap": fresh_target["accuracy"] - checkpoint_target["accuracy"], "cpu_latency_ms_batch": latency_ms(model, source_x, args.latency_repeats), "elapsed_seconds": time.time() - started, "scientific_conclusion_allowed": False}
            run_dir = output / name / f"seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            all_runs.append(result)
    summary = {"metadata": metadata, "runs": all_runs, "grouped": {name: [run for run in all_runs if run["architecture"] == name] for name in args.architectures}, "interpretation": "CPU smoke and architecture utility comparison only; not a formal LoP conclusion."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "architectures": args.architectures, "seeds": args.seeds, "runs": len(all_runs), "summary": str(output / "summary.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
