#!/usr/bin/env python3
"""Run a continuous subject-stream EEG architecture LoP pilot.

The protocol uses the public EdgeForge ISRUC medium development subset:
source subjects are trained first, target subjects are then adapted in a fixed
order, and a fixed retention set is evaluated after every probe budget.  At
each target stage the carried warm model is compared with a newly initialized
fresh model under the same target batches and finite adaptation budget.

This is a development/pilot runner.  It writes results outside the Git
checkout and always records ``scientific_conclusion_allowed=false``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import torch
import torch.nn.functional as F

from edgeforge.eeg_models import EEGModelConfig, build_eeg_decoder


DEFAULT_SOURCE_SUBJECTS = (1, 3, 4, 6, 7, 9, 10, 21)
DEFAULT_TARGET_SUBJECTS = (2, 11, 12, 13, 14, 15, 16, 17)
DEFAULT_RETENTION_SUBJECTS = (5, 18, 19, 20)
DEFAULT_BUDGETS = (0, 5, 10, 25, 50)


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def macro_f1(predictions: torch.Tensor, labels: torch.Tensor, classes: int) -> float:
    values: list[float] = []
    for cls in range(classes):
        tp = int(((predictions == cls) & (labels == cls)).sum().item())
        fp = int(((predictions == cls) & (labels != cls)).sum().item())
        fn = int(((predictions != cls) & (labels == cls)).sum().item())
        denominator = 2 * tp + fp + fn
        values.append(0.0 if denominator == 0 else (2.0 * tp / denominator))
    return float(np.mean(values))


def _subject_files(root: Path, group: str, subject: int) -> list[Path]:
    paths = sorted(
        (root / group / str(subject) / "data").glob("*.npy"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )
    if not paths:
        raise FileNotFoundError(f"no data files for {group} subject {subject} under {root}")
    return paths


def read_subject(root: Path, group: str, subject: int) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    values: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    files: list[dict[str, Any]] = []
    for data_path in _subject_files(root, group, subject):
        label_path = root / group / str(subject) / "label" / data_path.name
        data = np.load(data_path, allow_pickle=False)
        target = np.load(label_path, allow_pickle=False).reshape(-1)
        if data.shape != (20, 8, 3000) or target.shape != (20,):
            raise ValueError(f"invalid pair {data_path}: data={data.shape} labels={target.shape}")
        values.append(data.astype(np.float32, copy=False))
        labels.append(target.astype(np.int64, copy=False))
        files.append({
            "data": str(data_path),
            "label": str(label_path),
            "data_sha256": digest_file(data_path),
            "label_sha256": digest_file(label_path),
        })
    return (
        torch.from_numpy(np.concatenate(values, axis=0)),
        torch.from_numpy(np.concatenate(labels, axis=0)),
        files,
    )


def read_group(root: Path, group: str, subjects: Iterable[int]) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    values: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    files: list[dict[str, Any]] = []
    for subject in subjects:
        subject_values, subject_labels, subject_files = read_subject(root, group, int(subject))
        values.append(subject_values)
        labels.append(subject_labels)
        files.extend(subject_files)
    if not values:
        raise ValueError(f"empty {group} group")
    return torch.cat(values, dim=0), torch.cat(labels, dim=0), files


def read_target_stage(root: Path, subject: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    values, labels, files = read_subject(root, "target", int(subject))
    # The data card defines the chronological split for every 20-epoch file.
    train_values = values.reshape(-1, 20, 8, 3000)[:, :10].reshape(-1, 8, 3000)
    train_labels = labels.reshape(-1, 20)[:, :10].reshape(-1)
    eval_values = values.reshape(-1, 20, 8, 3000)[:, 10:].reshape(-1, 8, 3000)
    eval_labels = labels.reshape(-1, 20)[:, 10:].reshape(-1)
    return train_values, train_labels, eval_values, eval_labels, files


def model_config(name: str) -> EEGModelConfig:
    options: dict[str, Any] = {}
    if name == "lop_mlp":
        options = {"hidden_dims": [64, 64]}
    elif name == "brainuicl":
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


def build_model(name: str) -> torch.nn.Module:
    return build_eeg_decoder(model_config(name))


def make_batches(values: torch.Tensor, labels: torch.Tensor, batch_size: int, seed: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(int(values.shape[0]), generator=generator)
    return [
        (values[indexes], labels[indexes])
        for indexes in permutation.split(max(1, int(batch_size)))
    ]


@torch.no_grad()
def evaluate(model: torch.nn.Module, batches: list[tuple[torch.Tensor, torch.Tensor]], classes: int, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    predictions: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for values, target in batches:
        values = values.to(device)
        target = target.to(device)
        logits = model(values)
        losses.append(float(F.cross_entropy(logits, target).item()))
        predictions.append(logits.argmax(dim=-1).cpu())
        labels.append(target.cpu())
    if not losses:
        raise ValueError("evaluation received no batches")
    predicted = torch.cat(predictions)
    observed = torch.cat(labels)
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float((predicted == observed).float().mean().item()),
        "macro_f1": macro_f1(predicted, observed, classes),
    }


def _aulc(curve: list[dict[str, Any]], key: str) -> float:
    x = np.asarray([float(row["step"]) for row in curve], dtype=np.float64)
    y = np.asarray([float(row[key]) for row in curve], dtype=np.float64)
    if len(x) < 2 or float(x[-1]) <= 0.0:
        return float(y[-1]) if len(y) else 0.0
    integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(integrate(y, x) / x[-1])


def adapt_probe(
    initial: torch.nn.Module,
    train_batches: list[tuple[torch.Tensor, torch.Tensor]],
    eval_batches: list[tuple[torch.Tensor, torch.Tensor]],
    retention_batches: list[tuple[torch.Tensor, torch.Tensor]],
    budgets: tuple[int, ...],
    lr: float,
    classes: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    model = copy.deepcopy(initial).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    rows: list[dict[str, Any]] = []
    budget_set = set(int(item) for item in budgets)

    def record(step: int, train_loss: float | None) -> None:
        evaluation = evaluate(model, eval_batches, classes, device)
        retention = evaluate(model, retention_batches, classes, device)
        rows.append({
            "step": int(step),
            "train_loss": train_loss,
            "loss": evaluation["loss"],
            "accuracy": evaluation["accuracy"],
            "macro_f1": evaluation["macro_f1"],
            "retention_loss": retention["loss"],
            "retention_accuracy": retention["accuracy"],
            "retention_macro_f1": retention["macro_f1"],
        })

    record(0, None)
    iterator_index = 0
    for step in range(1, max(budgets) + 1):
        values, target = train_batches[iterator_index]
        iterator_index = (iterator_index + 1) % len(train_batches)
        model.train()
        values = values.to(device)
        target = target.to(device)
        loss = F.cross_entropy(model(values), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in budget_set:
            record(step, float(loss.detach().item()))

    summary = {
        "curve": rows,
        "budgets": [int(item) for item in budgets],
        "final": rows[-1],
        "accuracy_aulc": _aulc(rows, "accuracy"),
        "loss_aulc": _aulc(rows, "loss"),
        "macro_f1_aulc": _aulc(rows, "macro_f1"),
        "retention_accuracy_aulc": _aulc(rows, "retention_accuracy"),
        "retention_loss_aulc": _aulc(rows, "retention_loss"),
    }
    return model.cpu(), summary


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.data_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_x, source_y, source_files = read_group(root, "source", args.source_subjects)
    retention_x, retention_y, retention_files = read_group(root, "retention", args.retention_subjects)
    target_stages: list[dict[str, Any]] = []
    for subject in args.target_subjects:
        train_x, train_y, eval_x, eval_y, files = read_target_stage(root, int(subject))
        target_stages.append({
            "subject": int(subject),
            "train_x": train_x,
            "train_y": train_y,
            "eval_x": eval_x,
            "eval_y": eval_y,
            "files": files,
        })

    metadata = {
        "schema": "edgeforge.eeg-continuous-architecture-lop.v1",
        "dataset": "ISRUC",
        "data_root": str(root),
        "source_subjects": [int(item) for item in args.source_subjects],
        "target_subject_order": [int(item) for item in args.target_subjects],
        "retention_subjects": [int(item) for item in args.retention_subjects],
        "target_split": "first 10 epochs per file for adaptation; last 10 for held-out evaluation",
        "input_shape": [8, 3000],
        "classes": 5,
        "source_epochs": int(args.epochs),
        "source_lr": float(args.lr),
        "adapt_steps": int(max(args.budgets)),
        "budgets": [int(item) for item in args.budgets],
        "adapt_lr": float(args.adapt_lr),
        "batch_size": int(args.batch_size),
        "fresh_mode": "random-initialization-at-each-target-stage",
        "warm_mode": "carried-after-previous-stage-final-budget",
        "device": str(args.device),
        "scientific_conclusion_allowed": False,
        "source_files": source_files,
        "retention_files": retention_files,
    }

    all_runs: list[dict[str, Any]] = []
    for architecture_index, name in enumerate(args.architectures):
        for seed in args.seeds:
            seed_all(int(seed))
            started = time.time()
            warm = build_model(name)
            warm.to(args.device)
            source_batches = make_batches(source_x, source_y, args.batch_size, int(seed))
            source_optimizer = torch.optim.Adam(warm.parameters(), lr=float(args.lr))
            for _epoch in range(int(args.epochs)):
                warm.train()
                for values, target in source_batches:
                    values = values.to(args.device)
                    target = target.to(args.device)
                    loss = F.cross_entropy(warm(values), target)
                    source_optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(warm.parameters(), 1.0)
                    source_optimizer.step()
            warm = warm.cpu()
            source_metrics = evaluate(warm, make_batches(source_x, source_y, args.batch_size, 0), 5, torch.device("cpu"))
            retention_batches = make_batches(retention_x, retention_y, args.batch_size, int(seed) + 7000)
            stage_rows: list[dict[str, Any]] = []
            for stage_index, stage in enumerate(target_stages):
                stage_seed = int(seed) + 10000 + stage_index * 101 + architecture_index * 100000
                train_batches = make_batches(stage["train_x"], stage["train_y"], args.batch_size, stage_seed)
                eval_batches = make_batches(stage["eval_x"], stage["eval_y"], args.batch_size, stage_seed + 1)
                fresh_seed = int(seed) + 500000 + stage_index * 1009 + architecture_index * 100000
                seed_all(fresh_seed)
                fresh = build_model(name)
                warm_after, warm_probe = adapt_probe(warm, train_batches, eval_batches, retention_batches, args.budgets, args.adapt_lr, 5, torch.device("cpu"))
                fresh_after, fresh_probe = adapt_probe(fresh, train_batches, eval_batches, retention_batches, args.budgets, args.adapt_lr, 5, torch.device("cpu"))
                warm = warm_after
                gaps = []
                for warm_row, fresh_row in zip(warm_probe["curve"], fresh_probe["curve"]):
                    gaps.append({
                        "step": int(warm_row["step"]),
                        "fresh_gap": float(fresh_row["accuracy"] - warm_row["accuracy"]),
                        "fresh_loss_gap": float(warm_row["loss"] - fresh_row["loss"]),
                        "fresh_macro_f1_gap": float(fresh_row["macro_f1"] - warm_row["macro_f1"]),
                    })
                stage_rows.append({
                    "stage": int(stage_index),
                    "subject": int(stage["subject"]),
                    "files": stage["files"],
                    "train_sequences": int(stage["train_x"].shape[0] // 10),
                    "eval_sequences": int(stage["eval_x"].shape[0] // 10),
                    "warm": warm_probe,
                    "fresh": fresh_probe,
                    "gaps": gaps,
                    "fresh_gap_final": float(gaps[-1]["fresh_gap"]),
                    "fresh_loss_gap_final": float(gaps[-1]["fresh_loss_gap"]),
                    "fresh_auc_gap": float(fresh_probe["accuracy_aulc"] - warm_probe["accuracy_aulc"]),
                    "fresh_loss_auc_gap": float(warm_probe["loss_aulc"] - fresh_probe["loss_aulc"]),
                })
            result = {
                "architecture": name,
                "seed": int(seed),
                "parameters": int(sum(parameter.numel() for parameter in build_model(name).parameters())),
                "source": source_metrics,
                "stages": stage_rows,
                "elapsed_seconds": float(time.time() - started),
                "scientific_conclusion_allowed": False,
            }
            run_dir = output / name / f"seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run.json").write_text(json.dumps(_json_safe(result), indent=2) + "\n", encoding="utf-8")
            all_runs.append(result)

    summary = {
        "metadata": metadata,
        "runs": all_runs,
        "architectures": [str(item) for item in args.architectures],
        "seeds": [int(item) for item in args.seeds],
        "interpretation": "continuous architecture LoP pilot; not a formal scientific conclusion",
        "scientific_conclusion_allowed": False,
    }
    (output / "metadata.json").write_text(json.dumps(_json_safe(metadata), indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--architectures", nargs="+", default=["lop_mlp", "eegnet", "tcn", "transformer", "brainuicl"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[4321, 4322, 4323])
    parser.add_argument("--source-subjects", nargs="+", type=int, default=list(DEFAULT_SOURCE_SUBJECTS))
    parser.add_argument("--target-subjects", nargs="+", type=int, default=list(DEFAULT_TARGET_SUBJECTS))
    parser.add_argument("--retention-subjects", nargs="+", type=int, default=list(DEFAULT_RETENTION_SUBJECTS))
    parser.add_argument("--budgets", nargs="+", type=int, default=list(DEFAULT_BUDGETS))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--adapt-lr", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.budgets or args.budgets[0] != 0 or sorted(set(args.budgets)) != list(args.budgets):
        parser.error("--budgets must be sorted, unique and start with 0")
    if len(args.budgets) < 2 or max(args.budgets) <= 0:
        parser.error("--budgets must include at least one positive adaptation budget")
    return args


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(json.dumps({
        "status": "ok",
        "runs": len(summary["runs"]),
        "architectures": summary["architectures"],
        "seeds": summary["seeds"],
        "summary": str(args.output_root.resolve() / "summary.json"),
        "scientific_conclusion_allowed": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
