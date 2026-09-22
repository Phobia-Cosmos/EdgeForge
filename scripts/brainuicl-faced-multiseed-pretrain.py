#!/usr/bin/env python3
"""Reproducible FACED pretraining with a fixed subject partition.

FACED uses the aligned BrainUICL frontend.  This runner keeps the partition
seed fixed and varies only model/optimizer RNG, mirroring the ISRUC runner.
Checkpoints and logs are isolated per seed and are never written to the
external BrainUICL source tree.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


class _Tee:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "new_order": sorted(int(x) for x in new_idx),
        "split_seed": int(split_seed),
    }


def _run_seed(args, seed: int, subjects: list[int], imports: dict):
    from torch.utils.data import DataLoader

    build_blocks = imports["build_blocks"]
    evaluate = imports["evaluate"]
    flat_labels = imports["flat_labels"]
    flat_logits = imports["flat_logits"]
    forward_blocks = imports["forward_blocks"]
    make_loader = imports["make_loader"]
    save_blocks = imports["save_blocks"]
    set_train = imports["set_train"]
    fix_randomness = imports["fix_randomness"]

    run_root = args.output_root / args.dataset / f"seed{seed}"
    run_root.mkdir(parents=True, exist_ok=True)
    split = _fixed_split(subjects, args.split_seed)
    train_loader = make_loader(args.data_root, split["train_idx"], args.batch, False, args.num_worker)
    val_loader = make_loader(args.data_root, split["val_idx"], args.batch, False, args.num_worker)
    model_args = SimpleNamespace(**vars(args))
    model_args.seed = int(seed)
    model_args.dataset = "FACED"
    model_args.model_param = imports["ModelConfig"]("FACED")
    model_args.device = torch.device(
        f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    )
    fix_randomness(seed)
    blocks = build_blocks(model_args)
    optimizer = torch.optim.Adam(
        [parameter for block in blocks for parameter in block.parameters()],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    best_acc = -1.0
    best_epoch = 0
    best_blocks = None
    history = []
    started = time.time()
    with (run_root / "train.log").open("w", encoding="utf-8") as log_handle:
        with contextlib.redirect_stdout(_Tee(sys.stdout, log_handle)):
            print(json.dumps({"event": "pretrain_start", "dataset": "FACED", "seed": seed, "split_seed": args.split_seed, "device": str(model_args.device), "epochs": args.epochs}, sort_keys=True), flush=True)
            for epoch in range(1, args.epochs + 1):
                set_train(blocks, True)
                losses = []
                for eog, eeg, labels in train_loader:
                    logits = forward_blocks(blocks, eog.to(model_args.device), eeg.to(model_args.device), model_args)
                    loss = F.cross_entropy(flat_logits(logits), flat_labels(labels.to(model_args.device)))
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    for block in blocks:
                        torch.nn.utils.clip_grad_norm_(block.parameters(), 1.0)
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                validation = evaluate(blocks, val_loader, model_args)
                row = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "val_acc": validation["acc"], "val_mf1": validation["mf1"]}
                history.append(row)
                print(f"epoch={epoch}/{args.epochs} loss={row['train_loss']:.6f} val_acc={row['val_acc']:.6f} val_mf1={row['val_mf1']:.6f}", flush=True)
                if row["val_acc"] > best_acc:
                    best_acc = row["val_acc"]
                    best_epoch = epoch
                    best_blocks = copy.deepcopy(blocks)
    checkpoint_dir = run_root / "checkpoints" / "FACED" / "Pretrain"
    save_blocks(best_blocks, checkpoint_dir, seed)
    metrics = {
        "seed": seed,
        "split_seed": args.split_seed,
        "epochs": args.epochs,
        "elapsed_seconds": time.time() - started,
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
        "best_val_mf1_at_best_acc": history[best_epoch - 1]["val_mf1"] if best_epoch else None,
        "history": history,
        "checkpoints": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(checkpoint_dir.glob(f"*_{seed}.pkl"))
        ],
    }
    (run_root / "split.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    (run_root / "pretrain_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brainuicl-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--split-seed", type=int, default=4321)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    args = parser.parse_args()
    root = args.brainuicl_root.resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "experiments"))
    from regularization_cl_eeg import build_split
    from rttdp_brainuicl_full import build_blocks, evaluate, flat_labels, flat_logits, forward_blocks, make_loader, save_blocks, set_train, discover_subjects
    from utils.config import ModelConfig
    from utils.util import fix_randomness

    args.dataset = "FACED"
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    subjects = discover_subjects(args.data_root)
    if len(subjects) != 123:
        raise RuntimeError(f"Expected 123 FACED subjects, found {len(subjects)}")
    imports = {"build_blocks": build_blocks, "evaluate": evaluate, "flat_labels": flat_labels, "flat_logits": flat_logits, "forward_blocks": forward_blocks, "make_loader": make_loader, "save_blocks": save_blocks, "set_train": set_train, "fix_randomness": fix_randomness, "ModelConfig": ModelConfig}
    environment = {"python": sys.executable, "torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available()), "cuda": torch.version.cuda, "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], "data_root": str(args.data_root), "brainuicl_root": str(root)}
    (args.output_root / "FACED").mkdir(parents=True, exist_ok=True)
    (args.output_root / "FACED" / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    runs = []
    for seed in args.seeds:
        run_root = args.output_root / "FACED" / f"seed{seed}"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
        runs.append(_run_seed(args, int(seed), subjects, imports))
    manifest = {"schema": "edgeforge-brainuicl-faced-multiseed-pretrain-v1", "dataset": "FACED", "seeds": [int(x) for x in args.seeds], "split_seed": args.split_seed, "formal": args.epochs >= 100, "environment": environment, "runs": runs}
    (args.output_root / "FACED" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "dataset": "FACED", "seeds": args.seeds}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
