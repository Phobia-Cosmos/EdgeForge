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
import os
import pickle
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as F

from edgeforge.eeg_models import EEGModelConfig, build_eeg_decoder


DEFAULT_SOURCE_SUBJECTS = (1, 3, 4, 6, 7, 9, 10, 21)
DEFAULT_TARGET_SUBJECTS = (2, 11, 12, 13, 14, 15, 16, 17)
DEFAULT_RETENTION_SUBJECTS = (5, 18, 19, 20)
DEFAULT_BUDGETS = (0, 5, 10, 25, 50)
RETENTION_SUBSAMPLE_SEED = 20260904


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)


def stable_subject_seed(namespace: str, architecture: str, seed: int, subject: int) -> int:
    """Derive an order-independent seed without relying on Python's salted hash."""
    payload = f"{namespace}\0{architecture}\0{int(seed)}\0{int(subject)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def set_adaptation_train_mode(model: torch.nn.Module, freeze_batch_norm: bool) -> None:
    model.train()
    if freeze_batch_norm:
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()


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


def _read_subject_files(data_paths: Iterable[Path]) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    values: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    files: list[dict[str, Any]] = []
    for data_path in data_paths:
        label_path = data_path.parent.parent / "label" / data_path.name
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
    if not values:
        raise ValueError("cannot read an empty subject file selection")
    return (
        torch.from_numpy(np.concatenate(values, axis=0)),
        torch.from_numpy(np.concatenate(labels, axis=0)),
        files,
    )


def read_subject(root: Path, group: str, subject: int) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    return _read_subject_files(_subject_files(root, group, subject))


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


def read_source_split(
    root: Path,
    subjects: Iterable[int],
    eval_fraction: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Read source subjects and reserve the last file fraction per subject."""
    train_values: list[torch.Tensor] = []
    train_labels: list[torch.Tensor] = []
    eval_values: list[torch.Tensor] = []
    eval_labels: list[torch.Tensor] = []
    train_files: list[dict[str, Any]] = []
    eval_files: list[dict[str, Any]] = []
    for subject in subjects:
        paths = _subject_files(root, "source", int(subject))
        eval_count = 0
        if eval_fraction > 0.0:
            eval_count = max(1, int(np.ceil(len(paths) * eval_fraction)))
            if eval_count >= len(paths):
                raise ValueError(
                    f"source subject {subject} has {len(paths)} files, so "
                    f"--source-eval-fraction={eval_fraction} leaves no training file"
                )
        split_at = len(paths) - eval_count
        subject_train_x, subject_train_y, subject_train_files = _read_subject_files(paths[:split_at])
        train_values.append(subject_train_x)
        train_labels.append(subject_train_y)
        train_files.extend(subject_train_files)
        if eval_count:
            subject_eval_x, subject_eval_y, subject_eval_files = _read_subject_files(paths[split_at:])
            eval_values.append(subject_eval_x)
            eval_labels.append(subject_eval_y)
            eval_files.extend(subject_eval_files)
    if not train_values:
        raise ValueError("empty source group")
    source_x = torch.cat(train_values, dim=0)
    source_y = torch.cat(train_labels, dim=0)
    if eval_values:
        source_eval_x = torch.cat(eval_values, dim=0)
        source_eval_y = torch.cat(eval_labels, dim=0)
    else:
        # Fraction zero preserves the original training-set source evaluation.
        source_eval_x = source_x
        source_eval_y = source_y
    return source_x, source_y, source_eval_x, source_eval_y, train_files, eval_files


def limit_samples(
    values: torch.Tensor,
    labels: torch.Tensor,
    max_samples: int | None,
    seed: int = RETENTION_SUBSAMPLE_SEED,
) -> tuple[torch.Tensor, torch.Tensor, list[int] | None]:
    """Apply a fixed, reproducible sample cap while retaining original order."""
    if max_samples is None or max_samples >= int(values.shape[0]):
        return values, labels, None
    if max_samples <= 0:
        raise ValueError("max_samples must be positive when supplied")
    generator = np.random.default_rng(int(seed))
    selected = np.sort(generator.choice(int(values.shape[0]), size=int(max_samples), replace=False))
    indexes = torch.from_numpy(selected.astype(np.int64, copy=False))
    return values[indexes], labels[indexes], selected.tolist()


def validate_subject_roles(source: Iterable[int], target: Iterable[int], retention: Iterable[int]) -> None:
    role_values = {
        "source": [int(item) for item in source],
        "target": [int(item) for item in target],
        "retention": [int(item) for item in retention],
    }
    for role, values in role_values.items():
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate subject in {role} role: {values}")
    role_sets = {role: set(values) for role, values in role_values.items()}
    overlaps: list[str] = []
    roles = list(role_sets)
    for index, left in enumerate(roles):
        for right in roles[index + 1:]:
            shared = sorted(role_sets[left] & role_sets[right])
            if shared:
                overlaps.append(f"{left}/{right}={shared}")
    if overlaps:
        raise ValueError("subject roles must be mutually exclusive: " + ", ".join(overlaps))


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
    freeze_batch_norm: bool = False,
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
        set_adaptation_train_mode(model, freeze_batch_norm)
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


def _json_digest(value: Any) -> str:
    encoded = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(_json_safe(payload), indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_source_checkpoint(path: Path, expected_signature: str) -> dict[str, Any] | None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (FileNotFoundError, OSError, RuntimeError, EOFError, TypeError, ValueError, pickle.UnpicklingError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("status") != "complete" or payload.get("source_signature") != expected_signature:
        return None
    if not isinstance(payload.get("model_state_dict"), dict) or not isinstance(payload.get("source_metrics"), dict):
        return None
    return payload


def _file_fingerprints(files: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "data_sha256": str(item["data_sha256"]),
            "label_sha256": str(item["label_sha256"]),
        }
        for item in files
    ]


def _run_signature(experiment_signature: str, architecture: str, seed: int) -> str:
    return _json_digest({
        "experiment_signature": experiment_signature,
        "architecture": str(architecture),
        "seed": int(seed),
    })


def _stage_subjects_match(
    stages: Any,
    expected_subjects: Iterable[int] | None,
    *,
    require_complete: bool,
) -> bool:
    if not isinstance(stages, list):
        return False
    if expected_subjects is None:
        return True
    expected = [int(item) for item in expected_subjects]
    if len(stages) > len(expected) or (require_complete and len(stages) != len(expected)):
        return False
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            return False
        try:
            stage_index = int(stage.get("stage", -1))
            subject = int(stage.get("subject", -1))
        except (TypeError, ValueError):
            return False
        if stage_index != index or subject != expected[index]:
            return False
    return True


def _load_completed_run(
    path: Path,
    expected_signature: str,
    expected_subjects: Iterable[int] | None = None,
) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("status") != "complete" or payload.get("run_signature") != expected_signature:
        return None
    if not isinstance(payload.get("source"), dict):
        return None
    if not _stage_subjects_match(payload.get("stages"), expected_subjects, require_complete=True):
        return None
    return payload


def _load_partial_run(
    path: Path,
    expected_signature: str,
    expected_subjects: Iterable[int],
) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("status") != "partial" or payload.get("run_signature") != expected_signature:
        return None
    if not isinstance(payload.get("source"), dict):
        return None
    stages = payload.get("stages")
    if not _stage_subjects_match(stages, expected_subjects, require_complete=False) or not stages:
        return None
    checkpoint = payload.get("stage_checkpoint")
    if not isinstance(checkpoint, dict):
        return None
    try:
        completed_stages = int(checkpoint.get("completed_stages", -1))
    except (TypeError, ValueError):
        return None
    if completed_stages != len(stages):
        return None
    filename = checkpoint.get("file")
    if not isinstance(filename, str) or Path(filename).name != filename:
        return None
    return payload


def _load_stage_checkpoint(
    path: Path,
    expected_signature: str,
    expected_completed_stages: int,
) -> dict[str, Any] | None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (FileNotFoundError, OSError, RuntimeError, EOFError, TypeError, ValueError, pickle.UnpicklingError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("status") != "complete" or payload.get("run_signature") != expected_signature:
        return None
    try:
        completed_stages = int(payload.get("completed_stages", -1))
    except (TypeError, ValueError):
        return None
    if completed_stages != int(expected_completed_stages):
        return None
    if not isinstance(payload.get("model_state_dict"), dict):
        return None
    return payload


def _build_summary(
    metadata: dict[str, Any],
    completed: dict[tuple[str, int], dict[str, Any]],
    architectures: Iterable[str],
    seeds: Iterable[int],
) -> dict[str, Any]:
    architecture_list = [str(item) for item in architectures]
    seed_list = [int(item) for item in seeds]
    ordered_runs = [
        completed[(name, seed)]
        for name in architecture_list
        for seed in seed_list
        if (name, seed) in completed
    ]
    return {
        "metadata": metadata,
        "runs": ordered_runs,
        "architectures": architecture_list,
        "seeds": seed_list,
        "planned_runs": len(architecture_list) * len(seed_list),
        "completed_runs": len(ordered_runs),
        "interpretation": "continuous architecture LoP pilot; not a formal scientific conclusion",
        "scientific_conclusion_allowed": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.data_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_eval_fraction = float(getattr(args, "source_eval_fraction", 0.0))
    retention_max_samples = getattr(args, "retention_max_samples", None)
    resume = bool(getattr(args, "resume", False))
    freeze_batch_norm = bool(getattr(args, "freeze_batch_norm", False))
    source_checkpoint_argument = getattr(args, "source_checkpoint_root", None)
    source_checkpoint_root = Path(source_checkpoint_argument).resolve() if source_checkpoint_argument is not None else None
    if not 0.0 <= source_eval_fraction < 1.0:
        raise ValueError("source_eval_fraction must be in [0, 1)")
    if retention_max_samples is not None and int(retention_max_samples) <= 0:
        raise ValueError("retention_max_samples must be positive when supplied")
    validate_subject_roles(args.source_subjects, args.target_subjects, args.retention_subjects)
    device = torch.device(args.device)
    source_x, source_y, source_eval_x, source_eval_y, source_train_files, source_eval_files = read_source_split(
        root,
        args.source_subjects,
        source_eval_fraction,
    )
    retention_x, retention_y, retention_files = read_group(root, "retention", args.retention_subjects)
    retention_original_samples = int(retention_x.shape[0])
    retention_x, retention_y, retention_sample_indexes = limit_samples(
        retention_x,
        retention_y,
        retention_max_samples,
    )
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
        "source_eval_fraction": source_eval_fraction,
        "source_split": (
            "last ceil(files * source_eval_fraction) files per source subject are held out"
            if source_eval_fraction > 0.0
            else "disabled; source metrics reuse the source training set for backward compatibility"
        ),
        "source_train_samples": int(source_x.shape[0]),
        "source_eval_samples": int(source_eval_x.shape[0]),
        "target_split": "first 10 epochs per file for adaptation; last 10 for held-out evaluation",
        "input_shape": [8, 3000],
        "classes": 5,
        "source_epochs": int(args.epochs),
        "source_lr": float(args.lr),
        "source_batch_order": "single deterministic permutation repeated for every source epoch",
        "adapt_steps": int(max(args.budgets)),
        "budgets": [int(item) for item in args.budgets],
        "adapt_lr": float(args.adapt_lr),
        "batch_size": int(args.batch_size),
        "fresh_mode": "random-initialization-at-each-target-stage",
        "warm_mode": "carried-after-previous-stage-final-budget",
        "target_seed_scheme": "sha256(namespace, architecture, model_seed, target_subject); independent of stage/order",
        "freeze_batch_norm": freeze_batch_norm,
        "adapt_optimizer": "Adam",
        "adapt_optimizer_state_persisted": False,
        "device": str(device),
        "scientific_conclusion_allowed": False,
        "source_files": source_train_files + source_eval_files,
        "source_train_files": source_train_files,
        "source_eval_files": source_eval_files,
        "target_files": [
            {"subject": int(stage["subject"]), "files": stage["files"]}
            for stage in target_stages
        ],
        "retention_files": retention_files,
        "retention_original_samples": retention_original_samples,
        "retention_samples": int(retention_x.shape[0]),
        "retention_max_samples": None if retention_max_samples is None else int(retention_max_samples),
        "retention_sampling": {
            "method": "all" if retention_sample_indexes is None else "numpy-pcg64-without-replacement-sorted",
            "seed": None if retention_sample_indexes is None else RETENTION_SUBSAMPLE_SEED,
            "selected_indexes": retention_sample_indexes,
        },
        "source_checkpoint_root": None if source_checkpoint_root is None else str(source_checkpoint_root),
        "source_checkpoint_policy": "content-and-training-config signature; shared source weights and metrics",
    }
    metadata["experiment_signature"] = _json_digest(metadata)
    atomic_write_json(output / "metadata.json", metadata)

    completed: dict[tuple[str, int], dict[str, Any]] = {}
    summary = _build_summary(metadata, completed, args.architectures, args.seeds)
    atomic_write_json(output / "summary.json", summary)
    for name in args.architectures:
        for seed in args.seeds:
            key = (str(name), int(seed))
            run_dir = output / str(name) / f"seed{seed}"
            run_path = run_dir / "run.json"
            expected_signature = _run_signature(metadata["experiment_signature"], str(name), int(seed))
            resumed = (
                _load_completed_run(run_path, expected_signature, args.target_subjects)
                if resume
                else None
            )
            if resumed is not None:
                completed[key] = resumed
                summary = _build_summary(metadata, completed, args.architectures, args.seeds)
                atomic_write_json(output / "summary.json", summary)
                continue
            partial = (
                _load_partial_run(run_path, expected_signature, args.target_subjects)
                if resume
                else None
            )
            seed_all(int(seed))
            started = time.time()
            warm = build_model(str(name)).to(device)
            source_signature = _json_digest({
                "schema": "edgeforge.eeg-source-checkpoint.v1",
                "architecture": str(name),
                "model_config": model_config(str(name)).to_dict(),
                "seed": int(seed),
                "source_subjects": [int(item) for item in args.source_subjects],
                "source_eval_fraction": source_eval_fraction,
                "source_train_files": _file_fingerprints(source_train_files),
                "source_eval_files": _file_fingerprints(source_eval_files),
                "source_epochs": int(args.epochs),
                "source_lr": float(args.lr),
                "batch_size": int(args.batch_size),
                "batch_order": "single deterministic permutation repeated for every source epoch",
                "device": str(device),
                "torch_version": str(torch.__version__),
            })
            source_checkpoint_path = (
                source_checkpoint_root / str(name) / f"seed{seed}-{source_signature}.pt"
                if source_checkpoint_root is not None
                else None
            )
            source_checkpoint_hit = False
            source_state_origin = "trained"
            elapsed_before = 0.0
            stage_rows: list[dict[str, Any]] = []
            previous_stage_checkpoint: Path | None = None
            partial_stage_checkpoint = None
            if partial is not None:
                descriptor = partial["stage_checkpoint"]
                candidate_path = run_dir / str(descriptor["file"])
                partial_stage_checkpoint = _load_stage_checkpoint(
                    candidate_path,
                    expected_signature,
                    len(partial["stages"]),
                )
                if partial_stage_checkpoint is not None:
                    try:
                        warm.load_state_dict(partial_stage_checkpoint["model_state_dict"], strict=True)
                    except RuntimeError:
                        partial_stage_checkpoint = None
                    else:
                        stage_rows = partial["stages"]
                        source_metrics = partial["source"]
                        elapsed_before = float(partial.get("elapsed_seconds", 0.0))
                        previous_stage_checkpoint = candidate_path
                        source_state_origin = "partial-stage-checkpoint"

            source_checkpoint = None
            if partial_stage_checkpoint is None and source_checkpoint_path is not None:
                source_checkpoint = _load_source_checkpoint(source_checkpoint_path, source_signature)
            if partial_stage_checkpoint is None and source_checkpoint is not None:
                try:
                    warm.load_state_dict(source_checkpoint["model_state_dict"], strict=True)
                except RuntimeError:
                    source_checkpoint = None
                else:
                    source_metrics = source_checkpoint["source_metrics"]
                    source_checkpoint_hit = True
                    source_state_origin = "source-checkpoint"
            if partial_stage_checkpoint is None and source_checkpoint is None:
                source_batches = make_batches(source_x, source_y, args.batch_size, int(seed))
                source_optimizer = torch.optim.Adam(warm.parameters(), lr=float(args.lr))
                for _epoch in range(int(args.epochs)):
                    warm.train()
                    for values, target in source_batches:
                        values = values.to(device)
                        target = target.to(device)
                        loss = F.cross_entropy(warm(values), target)
                        source_optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(warm.parameters(), 1.0)
                        source_optimizer.step()
                source_metrics = evaluate(
                    warm,
                    make_batches(source_eval_x, source_eval_y, args.batch_size, 0),
                    5,
                    device,
                )
                if source_checkpoint_path is not None:
                    atomic_save_checkpoint(source_checkpoint_path, {
                        "schema": "edgeforge.eeg-source-checkpoint.v1",
                        "status": "complete",
                        "source_signature": source_signature,
                        "architecture": str(name),
                        "seed": int(seed),
                        "model_state_dict": {
                            key: value.detach().cpu()
                            for key, value in warm.state_dict().items()
                        },
                        "source_metrics": source_metrics,
                    })
            warm = warm.cpu()
            retention_batches = make_batches(retention_x, retention_y, args.batch_size, int(seed) + 7000)
            parameter_count = int(sum(parameter.numel() for parameter in warm.parameters()))
            for stage_index, stage in enumerate(target_stages[len(stage_rows):], start=len(stage_rows)):
                stage_seed = stable_subject_seed("target-train", str(name), int(seed), int(stage["subject"]))
                eval_seed = stable_subject_seed("target-eval", str(name), int(seed), int(stage["subject"]))
                train_batches = make_batches(stage["train_x"], stage["train_y"], args.batch_size, stage_seed)
                eval_batches = make_batches(stage["eval_x"], stage["eval_y"], args.batch_size, eval_seed)
                fresh_seed = stable_subject_seed("fresh-model", str(name), int(seed), int(stage["subject"]))
                seed_all(fresh_seed)
                fresh = build_model(str(name))
                warm_after, warm_probe = adapt_probe(
                    warm,
                    train_batches,
                    eval_batches,
                    retention_batches,
                    args.budgets,
                    args.adapt_lr,
                    5,
                    device,
                    freeze_batch_norm,
                )
                _, fresh_probe = adapt_probe(
                    fresh,
                    train_batches,
                    eval_batches,
                    retention_batches,
                    args.budgets,
                    args.adapt_lr,
                    5,
                    device,
                    freeze_batch_norm,
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
                    "train_batch_seed": int(stage_seed),
                    "eval_batch_seed": int(eval_seed),
                    "fresh_seed": int(fresh_seed),
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
                stage_checkpoint_path = run_dir / f"warm-stage-{len(stage_rows):04d}.pt"
                atomic_save_checkpoint(stage_checkpoint_path, {
                    "schema": "edgeforge.eeg-continuous-stage-checkpoint.v1",
                    "status": "complete",
                    "run_signature": expected_signature,
                    "completed_stages": len(stage_rows),
                    "model_state_dict": {
                        key: value.detach().cpu()
                        for key, value in warm.state_dict().items()
                    },
                })
                partial_result = {
                    "status": "partial",
                    "run_signature": expected_signature,
                    "experiment_signature": metadata["experiment_signature"],
                    "architecture": str(name),
                    "seed": int(seed),
                    "parameters": parameter_count,
                    "source": source_metrics,
                    "source_signature": source_signature,
                    "source_checkpoint": {
                        "path": None if source_checkpoint_path is None else str(source_checkpoint_path),
                        "cache_hit": source_checkpoint_hit,
                        "state_origin": source_state_origin,
                    },
                    "stages": stage_rows,
                    "stage_checkpoint": {
                        "file": stage_checkpoint_path.name,
                        "completed_stages": len(stage_rows),
                    },
                    "elapsed_seconds": elapsed_before + float(time.time() - started),
                    "scientific_conclusion_allowed": False,
                }
                atomic_write_json(run_path, partial_result)
                if previous_stage_checkpoint is not None and previous_stage_checkpoint != stage_checkpoint_path:
                    previous_stage_checkpoint.unlink(missing_ok=True)
                previous_stage_checkpoint = stage_checkpoint_path
            result = {
                "status": "complete",
                "run_signature": expected_signature,
                "experiment_signature": metadata["experiment_signature"],
                "architecture": str(name),
                "seed": int(seed),
                "parameters": parameter_count,
                "source": source_metrics,
                "source_signature": source_signature,
                "source_checkpoint": {
                    "path": None if source_checkpoint_path is None else str(source_checkpoint_path),
                    "cache_hit": source_checkpoint_hit,
                    "state_origin": source_state_origin,
                },
                "stages": stage_rows,
                "elapsed_seconds": elapsed_before + float(time.time() - started),
                "scientific_conclusion_allowed": False,
            }
            atomic_write_json(run_path, result)
            if previous_stage_checkpoint is not None:
                previous_stage_checkpoint.unlink(missing_ok=True)
            completed[key] = result
            summary = _build_summary(metadata, completed, args.architectures, args.seeds)
            atomic_write_json(output / "summary.json", summary)
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
    parser.add_argument(
        "--source-eval-fraction",
        type=float,
        default=0.0,
        help="Hold out this final fraction of files within every source subject; 0 preserves training-set evaluation.",
    )
    parser.add_argument(
        "--retention-max-samples",
        type=int,
        default=None,
        help="Deterministically cap the fixed retention set to this many epochs.",
    )
    parser.add_argument(
        "--source-checkpoint-root",
        type=Path,
        default=None,
        help="Optional shared cache for source-trained architecture/seed checkpoints.",
    )
    parser.add_argument("--budgets", nargs="+", type=int, default=list(DEFAULT_BUDGETS))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--adapt-lr", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--freeze-batch-norm",
        action="store_true",
        help="Freeze BatchNorm running statistics during warm/fresh adaptation probes.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete architecture/seed run.json files whose experiment signatures match.",
    )
    args = parser.parse_args()
    if not args.budgets or args.budgets[0] != 0 or sorted(set(args.budgets)) != list(args.budgets):
        parser.error("--budgets must be sorted, unique and start with 0")
    if len(args.budgets) < 2 or max(args.budgets) <= 0:
        parser.error("--budgets must include at least one positive adaptation budget")
    if not 0.0 <= args.source_eval_fraction < 1.0:
        parser.error("--source-eval-fraction must be in [0, 1)")
    if args.retention_max_samples is not None and args.retention_max_samples <= 0:
        parser.error("--retention-max-samples must be positive")
    try:
        validate_subject_roles(args.source_subjects, args.target_subjects, args.retention_subjects)
    except ValueError as error:
        parser.error(str(error))
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
