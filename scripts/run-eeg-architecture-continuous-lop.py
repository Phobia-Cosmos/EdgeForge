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

from edgeforge import lop_metrics

from edgeforge.eeg_models import EEGModelConfig, build_eeg_decoder


DEFAULT_SOURCE_SUBJECTS = (1, 3, 4, 6, 7, 9, 10, 21)
DEFAULT_TARGET_SUBJECTS = (2, 11, 12, 13, 14, 15, 16, 17)
DEFAULT_RETENTION_SUBJECTS = (5, 18, 19, 20)
DEFAULT_BUDGETS = (0, 5, 10, 25, 50)
ADAPTATION_STRATEGIES = ("plain", "source_replay", "l2_sp", "replay_l2_sp")
INPUT_NORMALIZATIONS = ("none", "epoch_rms", "channel_zscore")


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


def normalize_inputs(values: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply deterministic label-preserving normalization independently per epoch."""
    if mode not in INPUT_NORMALIZATIONS:
        raise ValueError(f"unknown input normalization: {mode}")
    if mode == "none":
        return values
    if values.ndim != 3:
        raise ValueError(f"expected [epochs, channels, samples], got {tuple(values.shape)}")
    values = values.float()
    if mode == "epoch_rms":
        scale = torch.sqrt(torch.mean(torch.square(values), dim=(1, 2), keepdim=True)).clamp_min(1e-12)
        return values / scale
    mean = values.mean(dim=-1, keepdim=True)
    scale = values.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-12)
    return (values - mean) / scale


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


def _gradient_probe(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    """Measure one deterministic target-loss gradient without updating state."""
    was_training = bool(model.training)
    model.eval()
    values, target = (item.to(device) for item in batch)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    try:
        logits = model(values)
        loss = F.cross_entropy(logits, target)
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        flattened = [gradient.detach().float().reshape(-1) for gradient in gradients if gradient is not None]
        if not flattened:
            return {"status": "unavailable", "reason": "model produced no parameter gradients"}
        vector = torch.cat(flattened).reshape(1, -1)
        summary = lop_metrics.gradient_summary(vector)
        return {
            "status": "computed",
            "loss": float(loss.detach().item()),
            "norm_l2": float(torch.linalg.vector_norm(vector).item()),
            "nonzero_fraction": float((vector.abs() > 1e-12).float().mean().item()),
            "summary": {
                "norm_mean": summary.get("norm_mean"),
                "norm_std": summary.get("norm_std"),
                "effective_rank": summary.get("effective_rank"),
            },
        }
    finally:
        model.train(was_training)


def checkpoint_diagnostics(
    model: torch.nn.Module,
    calibration_batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    *,
    reference: torch.nn.Module | None = None,
    max_observations: int = 256,
) -> dict[str, Any]:
    """Capture compact representation/state diagnostics at one probe point.

    These values are intentionally secondary evidence.  The LoP outcome is
    still the fixed-budget fresh-vs-warm gap, and the diagnostics never alter
    optimizer state or model parameters.
    """
    values, _target = (item.to(device) for item in calibration_batch)
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            bundle = model.forward_bundle(values) if hasattr(model, "forward_bundle") else None
        representations = {} if bundle is None else getattr(bundle, "representations", {})
        specs = model.representation_specs() if hasattr(model, "representation_specs") else {}
        layer_rows: dict[str, Any] = {}
        for name, representation in representations.items():
            if not isinstance(representation, torch.Tensor):
                continue
            spec = specs.get(name, {})
            axis = int(spec.get("feature_axis", -1))
            try:
                spectrum = lop_metrics.spectral_summary(
                    representation,
                    feature_axis=axis,
                    center=True,
                    max_observations=max(0, int(max_observations)),
                )
                activation = lop_metrics.activation_summary(representation, kind=str(spec.get("kind", "unknown")))
                layer_rows[name] = {
                    "shape": [int(item) for item in representation.shape],
                    "effective_rank": spectrum.get("effective_rank"),
                    "effective_rank_normalized": spectrum.get("effective_rank_normalized"),
                    "stable_rank": spectrum.get("stable_rank"),
                    "rank95": spectrum.get("rank95"),
                    "sigma_max": spectrum.get("sigma_max"),
                    "near_zero_fraction": activation.get("near_zero_fraction"),
                    "saturated_fraction": activation.get("saturated_fraction"),
                    "mean": activation.get("mean"),
                    "std": activation.get("std"),
                }
            except (RuntimeError, ValueError) as error:
                layer_rows[name] = {"status": "error", "error": f"{type(error).__name__}: {error}"}
        parameters = lop_metrics.parameter_norm_summary(model, reference)
        gradient = _gradient_probe(model, calibration_batch, device)
        return {
            "status": "computed",
            "calibration_batch_size": int(values.shape[0]),
            "max_observations": int(max_observations),
            "representations": layer_rows,
            "parameter_norm": {
                "global_l2": parameters.get("global_l2"),
                "global_delta_l2": parameters.get("global_delta_l2"),
                "global_relative_update": parameters.get("global_relative_update"),
                "parameter_count": parameters.get("parameter_count"),
            },
            "gradient": gradient,
            "interpretation": "descriptive checkpoint diagnostics; not a causal LoP outcome",
        }
    finally:
        model.train(was_training)


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
    *,
    adaptation_strategy: str = "plain",
    replay_batches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    replay_ratio: float = 0.25,
    l2_sp_lambda: float = 1e-4,
    checkpoint_diagnostics_enabled: bool = False,
    diagnostic_max_observations: int = 256,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if adaptation_strategy not in ADAPTATION_STRATEGIES:
        raise ValueError(f"unknown adaptation strategy: {adaptation_strategy}")
    uses_replay = adaptation_strategy in {"source_replay", "replay_l2_sp"}
    uses_l2_sp = adaptation_strategy in {"l2_sp", "replay_l2_sp"}
    if uses_replay and not replay_batches:
        raise ValueError("replay_batches are required for replay strategies")
    if not 0.0 <= float(replay_ratio) < 1.0:
        raise ValueError("replay_ratio must be in [0, 1)")
    if float(l2_sp_lambda) < 0.0:
        raise ValueError("l2_sp_lambda must be non-negative")
    model = copy.deepcopy(initial).to(device)
    reference = copy.deepcopy(initial).to(device) if checkpoint_diagnostics_enabled else None
    anchor = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    rows: list[dict[str, Any]] = []
    budget_set = set(int(item) for item in budgets)

    def record(step: int, train_loss: float | None) -> None:
        evaluation = evaluate(model, eval_batches, classes, device)
        retention = evaluate(model, retention_batches, classes, device)
        rows.append({
            "step": int(step),
            "train_loss": train_loss,
            "target_train_loss": None,
            "replay_train_loss": None,
            "data_loss": None,
            "l2_sp_penalty": None,
            "loss": evaluation["loss"],
            "accuracy": evaluation["accuracy"],
            "macro_f1": evaluation["macro_f1"],
            "retention_loss": retention["loss"],
            "retention_accuracy": retention["accuracy"],
            "retention_macro_f1": retention["macro_f1"],
        })
        if checkpoint_diagnostics_enabled:
            rows[-1]["diagnostics"] = checkpoint_diagnostics(
                model,
                eval_batches[0],
                device,
                reference=reference,
                max_observations=diagnostic_max_observations,
            )

    record(0, None)
    iterator_index = 0
    replay_iterator_index = 0
    for step in range(1, max(budgets) + 1):
        values, target = train_batches[iterator_index]
        iterator_index = (iterator_index + 1) % len(train_batches)
        model.train()
        values = values.to(device)
        target = target.to(device)
        target_loss = F.cross_entropy(model(values), target)
        replay_loss: torch.Tensor | None = None
        if uses_replay:
            replay_values, replay_target = replay_batches[replay_iterator_index]
            replay_iterator_index = (replay_iterator_index + 1) % len(replay_batches)
            replay_values = replay_values.to(device)
            replay_target = replay_target.to(device)
            replay_loss = F.cross_entropy(model(replay_values), replay_target)
            data_loss = (1.0 - float(replay_ratio)) * target_loss + float(replay_ratio) * replay_loss
        else:
            data_loss = target_loss
        l2_sp_penalty: torch.Tensor | None = None
        if uses_l2_sp:
            l2_sp_penalty = torch.zeros((), device=device)
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and name in anchor:
                    l2_sp_penalty = l2_sp_penalty + (parameter - anchor[name]).pow(2).sum()
            loss = data_loss + 0.5 * float(l2_sp_lambda) * l2_sp_penalty
        else:
            loss = data_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in budget_set:
            record(step, float(loss.detach().item()))
            rows[-1]["target_train_loss"] = float(target_loss.detach().item())
            rows[-1]["replay_train_loss"] = None if replay_loss is None else float(replay_loss.detach().item())
            rows[-1]["data_loss"] = float(data_loss.detach().item())
            rows[-1]["l2_sp_penalty"] = None if l2_sp_penalty is None else float(l2_sp_penalty.detach().item())

    summary = {
        "adaptation_strategy": adaptation_strategy,
        "replay_ratio": float(replay_ratio) if uses_replay else 0.0,
        "l2_sp_lambda": float(l2_sp_lambda) if uses_l2_sp else 0.0,
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
    source_x = normalize_inputs(source_x, args.input_normalization)
    retention_x = normalize_inputs(retention_x, args.input_normalization)
    target_stages: list[dict[str, Any]] = []
    for subject in args.target_subjects:
        train_x, train_y, eval_x, eval_y, files = read_target_stage(root, int(subject))
        train_x = normalize_inputs(train_x, args.input_normalization)
        eval_x = normalize_inputs(eval_x, args.input_normalization)
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
        "adaptation_strategy": str(args.adaptation_strategy),
        "input_normalization": str(args.input_normalization),
        "replay_ratio": float(args.replay_ratio) if args.adaptation_strategy in {"source_replay", "replay_l2_sp"} else 0.0,
        "replay_loss_mixture": "(1-replay_ratio)*target_cross_entropy + replay_ratio*source_cross_entropy",
        "replay_sampling": "fixed deterministic source batches shared by warm and fresh probes",
        "l2_sp_lambda": float(args.l2_sp_lambda) if args.adaptation_strategy in {"l2_sp", "replay_l2_sp"} else 0.0,
        "l2_sp_anchor": "probe-initial parameters; carried warm state for warm and random initialization for fresh",
        "checkpoint_diagnostics": bool(args.checkpoint_diagnostics),
        "diagnostic_max_observations": int(args.diagnostic_max_observations),
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
                replay_batches = None
                if args.adaptation_strategy in {"source_replay", "replay_l2_sp"}:
                    replay_batches = make_batches(source_x, source_y, args.batch_size, stage_seed + 2)
                fresh_seed = int(seed) + 500000 + stage_index * 1009 + architecture_index * 100000
                seed_all(fresh_seed)
                fresh = build_model(name)
                warm_after, warm_probe = adapt_probe(
                    warm,
                    train_batches,
                    eval_batches,
                    retention_batches,
                    args.budgets,
                    args.adapt_lr,
                    5,
                    torch.device(args.device),
                    adaptation_strategy=args.adaptation_strategy,
                    replay_batches=replay_batches,
                    replay_ratio=args.replay_ratio,
                    l2_sp_lambda=args.l2_sp_lambda,
                    checkpoint_diagnostics_enabled=args.checkpoint_diagnostics,
                    diagnostic_max_observations=args.diagnostic_max_observations,
                )
                fresh_after, fresh_probe = adapt_probe(
                    fresh,
                    train_batches,
                    eval_batches,
                    retention_batches,
                    args.budgets,
                    args.adapt_lr,
                    5,
                    torch.device(args.device),
                    adaptation_strategy=args.adaptation_strategy,
                    replay_batches=replay_batches,
                    replay_ratio=args.replay_ratio,
                    l2_sp_lambda=args.l2_sp_lambda,
                    checkpoint_diagnostics_enabled=args.checkpoint_diagnostics,
                    diagnostic_max_observations=args.diagnostic_max_observations,
                )
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
                "adaptation_strategy": str(args.adaptation_strategy),
                "input_normalization": str(args.input_normalization),
                "replay_ratio": float(args.replay_ratio) if args.adaptation_strategy in {"source_replay", "replay_l2_sp"} else 0.0,
                "l2_sp_lambda": float(args.l2_sp_lambda) if args.adaptation_strategy in {"l2_sp", "replay_l2_sp"} else 0.0,
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
        "adaptation_strategy": str(args.adaptation_strategy),
        "input_normalization": str(args.input_normalization),
        "replay_ratio": float(args.replay_ratio) if args.adaptation_strategy in {"source_replay", "replay_l2_sp"} else 0.0,
        "l2_sp_lambda": float(args.l2_sp_lambda) if args.adaptation_strategy in {"l2_sp", "replay_l2_sp"} else 0.0,
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
    parser.add_argument("--adaptation-strategy", choices=ADAPTATION_STRATEGIES, default="plain")
    parser.add_argument("--input-normalization", choices=INPUT_NORMALIZATIONS, default="none", help="per-epoch input normalization; labels and epoch boundaries are unchanged")
    parser.add_argument("--replay-ratio", type=float, default=0.25, help="source replay weight in the target/source cross-entropy mixture")
    parser.add_argument("--l2-sp-lambda", type=float, default=1e-4, help="weight on half the squared distance from probe-initial parameters")
    parser.add_argument("--checkpoint-diagnostics", action="store_true", help="record compact representation, gradient and parameter probes at every budget")
    parser.add_argument("--diagnostic-max-observations", type=int, default=256)
    args = parser.parse_args()
    if not args.budgets or args.budgets[0] != 0 or sorted(set(args.budgets)) != list(args.budgets):
        parser.error("--budgets must be sorted, unique and start with 0")
    if len(args.budgets) < 2 or max(args.budgets) <= 0:
        parser.error("--budgets must include at least one positive adaptation budget")
    if args.diagnostic_max_observations < 0:
        parser.error("--diagnostic-max-observations must be non-negative")
    if not 0.0 <= args.replay_ratio < 1.0:
        parser.error("--replay-ratio must be in [0, 1)")
    if args.l2_sp_lambda < 0.0:
        parser.error("--l2-sp-lambda must be non-negative")
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
