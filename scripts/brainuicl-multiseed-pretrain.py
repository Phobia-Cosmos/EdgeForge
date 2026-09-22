#!/usr/bin/env python3
"""Run reproducible BrainUICL ISRUC pretraining for independent seeds.

The upstream BrainUICL entry point uses the model seed for both subject
sampling and parameter initialisation and writes checkpoints relative to the
current directory.  This EdgeForge-side runner keeps the subject split fixed
(``--split-seed``), varies the real training seed, and gives every seed an
isolated output directory.  It does not modify the external BrainUICL tree.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader


def _add_brainuicl_to_path(root: Path) -> None:
    root = root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _isruc_paths(data_root: Path) -> tuple[list[int], dict[int, tuple[list[str], list[str]]]]:
    path: list[int] = []
    path_name: dict[int, tuple[list[str], list[str]]] = {}
    for subject in range(1, 101):
        if subject in (8, 40):
            continue
        data_dir = data_root / str(subject) / "data"
        label_dir = data_root / str(subject) / "label"
        if not data_dir.is_dir() or not label_dir.is_dir():
            continue
        data: list[str] = []
        labels: list[str] = []
        index = 0
        while (data_dir / f"{index}.npy").exists() and (label_dir / f"{index}.npy").exists():
            data.append(str(data_dir / f"{index}.npy"))
            labels.append(str(label_dir / f"{index}.npy"))
            index += 1
        if data:
            path.append(subject)
            path_name[subject] = (data, labels)
    if len(path) < 5:
        raise ValueError(f"Need at least 5 ISRUC subjects; found {len(path)} under {data_root}")
    return path, path_name


def _fixed_split(subjects: list[int], split_seed: int) -> dict[str, list[int]]:
    rng = np.random.RandomState(split_seed)
    old_count = max(1, int(len(subjects) * 0.2))
    new_count = max(1, int(len(subjects) * 0.5))
    new_count = min(new_count, len(subjects) - old_count - 2)
    old_idx = list(rng.choice(subjects, old_count, replace=False))
    remaining = sorted(set(subjects) - set(old_idx))
    new_idx = list(rng.choice(remaining, new_count, replace=False))
    train_val = sorted(set(subjects) - set(old_idx) - set(new_idx))
    train_count = max(1, int(len(train_val) * 0.8))
    train_count = min(train_count, len(train_val) - 1)
    train_idx = list(rng.choice(train_val, train_count, replace=False))
    val_idx = [subject for subject in train_val if subject not in train_idx]
    return {
        "train_idx": sorted(int(x) for x in train_idx),
        "val_idx": sorted(int(x) for x in val_idx),
        "old_idx": sorted(int(x) for x in old_idx),
        "new_idx": sorted(int(x) for x in new_idx),
        "split_seed": int(split_seed),
    }


def _join(paths: dict[int, tuple[list[str], list[str]]], subjects: list[int]) -> list[list[str]]:
    data: list[str] = []
    labels: list[str] = []
    for subject in subjects:
        data.extend(paths[subject][0])
        labels.extend(paths[subject][1])
    return [data, labels]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _Tee:
    """Mirror a seed's stdout to the terminal and its immutable log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _run_seed(args: argparse.Namespace, seed: int, subjects, paths) -> dict:
    from dataloader.data_loader import Builder
    from trainer.pretrainer import pretraining
    from utils.util import fix_randomness

    run_root = args.output_root / args.dataset / f"seed{seed}"
    run_root.mkdir(parents=True, exist_ok=True)
    split = _fixed_split(subjects, args.split_seed)
    train_path = _join(paths, split["train_idx"])
    val_path = _join(paths, split["val_idx"])

    # Build loaders before reseeding the model.  The split remains identical
    # across all training seeds while model/optimizer initialisation differs.
    model_args = SimpleNamespace(
        dataset=args.dataset,
        gpu=args.gpu,
        seed=int(seed),
        batch=args.batch,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
        pretrain_epoch=args.epochs,
    )
    train_dataset = Builder(train_path, model_args).Dataset
    val_dataset = Builder(val_path, model_args).Dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_worker,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_worker,
    )
    fix_randomness(seed)
    previous_cwd = Path.cwd()
    started = time.time()
    try:
        os.chdir(run_root)
        with (run_root / "train.log").open("w", encoding="utf-8") as log_handle:
            with contextlib.redirect_stdout(_Tee(sys.stdout, log_handle)):
                print(
                    json.dumps(
                        {
                            "event": "pretrain_start",
                            "dataset": args.dataset,
                            "seed": seed,
                            "split_seed": args.split_seed,
                            "train_subjects": split["train_idx"],
                            "val_subjects": split["val_idx"],
                            "device": str(torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")),
                            "epochs": args.epochs,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                pretraining(train_loader, val_loader, model_args)
    finally:
        os.chdir(previous_cwd)

    checkpoint_dir = run_root / "model_parameter" / args.dataset / "Pretrain"
    files = sorted(checkpoint_dir.glob(f"*_{seed}.pkl"))
    if len(files) != 3:
        raise RuntimeError(f"Expected three checkpoints for seed {seed}, found {files}")
    elapsed = time.time() - started
    log_text = (run_root / "train.log").read_text(encoding="utf-8") if (run_root / "train.log").exists() else ""
    acc = [float(x) for x in re.findall(r"Accuracy on sleep:\s*([0-9.eE+-]+)", log_text)]
    mf1 = [float(x) for x in re.findall(r"F1 score on sleep:\s*([0-9.eE+-]+)", log_text)]
    metrics = {
        "seed": seed,
        "split_seed": args.split_seed,
        "epochs": args.epochs,
        "elapsed_seconds": elapsed,
        "val_acc_history": acc,
        "val_mf1_history": mf1,
        "best_val_acc": max(acc) if acc else None,
        "best_val_mf1_at_best_acc": mf1[acc.index(max(acc))] if acc else None,
        "checkpoints": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in files
        ],
    }
    (run_root / "split.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    (run_root / "pretrain_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brainuicl-root", type=Path, required=True)
    parser.add_argument("--dataset", default="ISRUC")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--split-seed", type=int, default=4321)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    args = parser.parse_args()
    if args.dataset != "ISRUC":
        raise ValueError("This runner currently supports ISRUC; use pretrain_faced_aligned.py for FACED")
    _add_brainuicl_to_path(args.brainuicl_root)
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    subjects, paths = _isruc_paths(args.data_root)
    environment = {
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "dataset_root": str(args.data_root),
        "brainuicl_root": str(args.brainuicl_root.resolve()),
    }
    (args.output_root / args.dataset / "environment.json").parent.mkdir(parents=True, exist_ok=True)
    (args.output_root / args.dataset / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    all_metrics = []
    for seed in args.seeds:
        run_root = args.output_root / args.dataset / f"seed{seed}"
        run_root.mkdir(parents=True, exist_ok=True)
        command = " ".join(sys.argv)
        (run_root / "command.txt").write_text(command + "\n", encoding="utf-8")
        metrics = _run_seed(args, int(seed), subjects, paths)
        all_metrics.append(metrics)
    manifest = {
        "schema": "edgeforge-brainuicl-multiseed-pretrain-v1",
        "dataset": args.dataset,
        "seeds": [int(seed) for seed in args.seeds],
        "split_seed": args.split_seed,
        "formal": args.epochs >= 100,
        "environment": environment,
        "runs": all_metrics,
    }
    (args.output_root / args.dataset / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "dataset": args.dataset, "seeds": args.seeds}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
