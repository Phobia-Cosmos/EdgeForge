#!/usr/bin/env python3
"""Small EEG-aligned Transformer LoP diagnostic.

The input contract stays close to BrainUICL ISRUC: each file contains
``(20, 8, 3000)`` float32 windows and five sleep-stage labels.  A fixed,
non-learned time/frequency frontend produces one feature vector per epoch;
the learnable body is a two-layer, four-head Transformer.  The experiment is
an explicitly labelled supervised-oracle diagnostic: target labels are used
for the controlled adaptation update and only for held-out evaluation.  It is
not an online TTA or a final LoP claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def spectral_features(values: np.ndarray, sample_rate: float = 200.0) -> np.ndarray:
    """Return per-epoch features with time statistics and EEG bands."""
    if values.shape != (20, 8, 3000):
        raise ValueError(f"expected (20,8,3000), got {values.shape}")
    x = values.astype(np.float32, copy=False)
    time_mean = x.mean(axis=-1)
    time_std = x.std(axis=-1)
    time_rms = np.sqrt(np.mean(x * x, axis=-1) + 1e-8)
    centered = x - time_mean[..., None]
    spectrum = np.abs(np.fft.rfft(centered, axis=-1)) ** 2 / x.shape[-1]
    freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / sample_rate)
    bands = ((0.5, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 30.0), (30.0, 45.0))
    powers = []
    total = spectrum[:, :, (freqs >= 0.5) & (freqs <= 45.0)].mean(axis=-1) + 1e-8
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        powers.append(np.log1p(spectrum[:, :, mask].mean(axis=-1) / total))
    return np.concatenate([time_mean, time_std, time_rms, *powers], axis=1).astype(np.float32)


class FileDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]], mean=None, std=None):
        self.records = []
        for data_path, label_path in pairs:
            features = spectral_features(np.load(data_path, allow_pickle=False))
            labels = np.load(label_path, allow_pickle=False).astype(np.int64).reshape(-1)
            if labels.shape != (20,):
                raise ValueError(f"expected 20 labels in {label_path}, got {labels.shape}")
            self.records.append((features, labels, data_path, label_path))
        if not self.records:
            raise ValueError("empty sequence file set")
        self.mean = np.asarray(mean if mean is not None else np.mean(np.concatenate([r[0] for r in self.records]), axis=(0, 1)), dtype=np.float32)
        self.std = np.asarray(std if std is not None else np.std(np.concatenate([r[0] for r in self.records]), axis=(0, 1)), dtype=np.float32)
        self.std = np.maximum(self.std, 1e-5)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        features, labels, data_path, label_path = self.records[index]
        return ((torch.from_numpy((features - self.mean) / self.std), torch.from_numpy(labels)), str(data_path), str(label_path))


class EEGTransformer(nn.Module):
    def __init__(self, input_dim: int = 64, d_model: int = 64, heads: int = 4, layers: int = 2, classes: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.projection = nn.Linear(input_dim, d_model)
        self.position = nn.Parameter(torch.zeros(1, 20, d_model))
        nn.init.normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "norm1": nn.LayerNorm(d_model),
                "attention": nn.MultiheadAttention(d_model, heads, dropout=0.0, batch_first=True),
                "norm2": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)),
            }) for _ in range(layers)
        ])
        self.classifier = nn.Linear(d_model, classes)

    def forward(self, x, return_diagnostics: bool = False):
        hidden = self.projection(x) + self.position
        attentions = []
        for block in self.blocks:
            residual = hidden
            query = block["norm1"](hidden)
            attended, weights = block["attention"](query, query, query, need_weights=True, average_attn_weights=False)
            hidden = residual + attended
            hidden = hidden + block["ffn"](block["norm2"](hidden))
            if return_diagnostics:
                attentions.append(weights.detach())
        logits = self.classifier(hidden)
        return (logits, hidden, attentions) if return_diagnostics else logits


def discover_pairs(root: Path, subjects: list[int], max_files: int) -> list[tuple[Path, Path]]:
    pairs = []
    for subject in subjects:
        subject_dir = root / str(subject)
        data_dir, label_dir = subject_dir / "data", subject_dir / "label"
        files = sorted(data_dir.glob("*.npy"), key=lambda p: int(p.stem))[:max_files]
        for data_path in files:
            label_path = label_dir / data_path.name
            if label_path.is_file():
                pairs.append((data_path, label_path))
    return pairs


def merge_stats(pairs: list[tuple[Path, Path]]) -> tuple[np.ndarray, np.ndarray]:
    arrays = [spectral_features(np.load(data, allow_pickle=False)) for data, _ in pairs]
    all_features = np.concatenate(arrays, axis=0)
    return all_features.mean(axis=0), np.maximum(all_features.std(axis=0), 1e-5)


def loader(pairs, mean, std, batch_size, shuffle, seed):
    dataset = FileDataset(pairs, mean, std)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator)


def flatten_logits(logits, labels):
    return logits.reshape(-1, logits.shape[-1]), labels.reshape(-1).long()


@torch.no_grad()
def evaluate(model, data_loader, device, diagnostics=False):
    model.eval()
    losses, correct, count = [], 0, 0
    hidden_rows, attention_rows = [], []
    for (features, labels), *_ in data_loader:
        features, labels = features.to(device), labels.to(device)
        output = model(features, return_diagnostics=diagnostics)
        logits, hidden, attention = output if diagnostics else (output, None, None)
        flat, target = flatten_logits(logits, labels)
        losses.append(float(F.cross_entropy(flat, target).item()))
        correct += int((flat.argmax(1) == target).sum())
        count += target.numel()
        if diagnostics:
            hidden_rows.append(hidden.detach().cpu())
            attention_rows.extend(attention)
    accuracy = correct / max(1, count)
    return {"loss": float(np.mean(losses)), "acc": accuracy, "hidden": hidden_rows, "attention": attention_rows}


def representation_metrics(evaluated: dict) -> dict:
    hidden = torch.cat(evaluated["hidden"], dim=0).reshape(-1, evaluated["hidden"][0].shape[-1]).float()
    hidden = hidden - hidden.mean(0, keepdim=True)
    singular = torch.linalg.svdvals(hidden)
    energy = singular.square()
    effective_rank = float(torch.exp(-(energy / energy.sum().clamp_min(1e-12)) .mul((energy / energy.sum().clamp_min(1e-12)).clamp_min(1e-12).log()).sum()))
    stable_rank = float((singular.square().sum() / singular.max().square().clamp_min(1e-12)).item())
    entropy_values = []
    for weights in evaluated["attention"]:
        probabilities = weights.float().clamp_min(1e-12)
        entropy_values.append(float((-(probabilities * probabilities.log()).sum(-1).mean()).item()))
    return {"effective_rank": effective_rank, "stable_rank": stable_rank, "attention_entropy": float(np.mean(entropy_values)) if entropy_values else None, "hidden_dim": int(hidden.shape[-1])}


def adapt(model, data_loader, eval_loader, retention_loader, device, steps, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    iterator = iter(data_loader)
    curve = []
    for step in range(steps + 1):
        if step == 0 or step in (5, 10, 25, steps):
            target = evaluate(model, eval_loader, device, diagnostics=step == steps)
            row = {"step": step, "loss": target["loss"], "acc": target["acc"]}
            if retention_loader is not None:
                retained = evaluate(model, retention_loader, device)
                row.update({"retention_acc": retained["acc"], "retention_loss": retained["loss"]})
            if step == steps:
                row["representation"] = representation_metrics(target)
            curve.append(row)
        if step == steps:
            break
        try:
            (features, labels), *_ = next(iterator)
        except StopIteration:
            iterator = iter(data_loader)
            (features, labels), *_ = next(iterator)
        model.train()
        features, labels = features.to(device), labels.to(device)
        flat, target = flatten_logits(model(features), labels)
        loss = F.cross_entropy(flat, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return curve


def pearson(xs, ys):
    if len(xs) < 3 or np.std(xs) == 0 or np.std(ys) == 0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[4321, 4322, 4323])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--adapt-steps", type=int, default=25)
    parser.add_argument("--max-files", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--adapt-lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    # Fixed partition: source subjects are used for pretraining; three held-out
    # subjects form the sequential target stream; subject 2 is retention.
    source_subjects = [1, 3, 4, 5, 6, 7]
    target_subjects = [9, 10, 11]
    retention_subject = 2
    source_pairs = discover_pairs(args.data_root, source_subjects, args.max_files)
    if len(source_pairs) < 6:
        raise RuntimeError(f"insufficient source files: {len(source_pairs)}")
    target_pairs = {subject: discover_pairs(args.data_root, [subject], args.max_files) for subject in target_subjects}
    retention_pairs = discover_pairs(args.data_root, [retention_subject], args.max_files)
    mean, std = merge_stats(source_pairs)
    metadata = {"protocol": "simple-transformer-eeg-lop-v1", "dataset": "ISRUC", "source_subjects": source_subjects, "target_subjects": target_subjects, "retention_subject": retention_subject, "input_shape": [20, 8, 3000], "feature_dim": 64, "frontend": "time_mean/std/rms + delta/theta/alpha/beta/gamma log power", "transformer": {"layers": 2, "heads": 4, "d_model": 64}, "supervised_oracle": True, "scientific_conclusion_allowed": False, "device": str(device), "torch": torch.__version__, "cuda": torch.version.cuda}
    (args.output_root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    all_runs = []
    for seed in args.seeds:
        started = time.time()
        seed_all(seed)
        run_dir = args.output_root / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        train_loader = loader(source_pairs, mean, std, args.batch_size, True, seed)
        source_eval_loader = loader(source_pairs, mean, std, args.batch_size, False, seed)
        model = EEGTransformer().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        pretrain_history = []
        for epoch in range(1, args.epochs + 1):
            model.train(); losses = []
            for (features, labels), *_ in train_loader:
                features, labels = features.to(device), labels.to(device)
                flat, target = flatten_logits(model(features), labels)
                loss = F.cross_entropy(flat, target)
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); losses.append(float(loss.detach().cpu()))
            evaluation = evaluate(model, source_eval_loader, device)
            pretrain_history.append({"epoch": epoch, "loss": float(np.mean(losses)), "acc": evaluation["acc"]})
        torch.save(model.state_dict(), run_dir / "pretrain.pt")
        stages = []
        current = copy.deepcopy(model)
        for stage, subject in enumerate(target_subjects, start=1):
            pairs = target_pairs[subject]
            if len(pairs) < 2:
                raise RuntimeError(f"target subject {subject} has fewer than two files")
            pivot = max(1, len(pairs) // 2)
            target_train = loader(pairs[:pivot], mean, std, args.batch_size, True, seed + stage)
            target_eval = loader(pairs[pivot:], mean, std, args.batch_size, False, seed + stage)
            retained = loader(retention_pairs, mean, std, args.batch_size, False, seed + stage)
            checkpoint_model = copy.deepcopy(current)
            fresh_model = EEGTransformer().to(device)
            checkpoint_curve = adapt(checkpoint_model, target_train, target_eval, retained, device, args.adapt_steps, args.adapt_lr)
            fresh_curve = adapt(fresh_model, target_train, target_eval, retained, device, args.adapt_steps, args.adapt_lr)
            final_checkpoint = checkpoint_curve[-1]; final_fresh = fresh_curve[-1]
            stage_result = {"stage": stage, "subject": subject, "train_files": len(pairs[:pivot]), "eval_files": len(pairs[pivot:]), "checkpoint": checkpoint_curve, "fresh": fresh_curve, "fresh_gap": final_fresh["acc"] - final_checkpoint["acc"], "retention_acc_drop": checkpoint_curve[0].get("retention_acc", 0.0) - final_checkpoint.get("retention_acc", 0.0), "effective_rank": final_checkpoint["representation"]["effective_rank"], "stable_rank": final_checkpoint["representation"]["stable_rank"], "attention_entropy": final_checkpoint["representation"]["attention_entropy"]}
            stages.append(stage_result)
            torch.save(checkpoint_model.state_dict(), run_dir / f"stage{stage}.pt")
            current = checkpoint_model
        run = {"seed": seed, "pretrain": pretrain_history, "stages": stages, "elapsed_seconds": time.time() - started, "scientific_conclusion_allowed": False}
        (run_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
        all_runs.append(run)
    pairs_for_corr = [(stage["effective_rank"], stage["fresh_gap"]) for run in all_runs for stage in run["stages"]]
    analysis = {"protocol": "lagged-descriptive-v1", "points": len(pairs_for_corr), "effective_rank_to_fresh_gap_pearson": pearson([x for x, _ in pairs_for_corr], [y for _, y in pairs_for_corr]), "interpretation": "descriptive only; seed-cluster bootstrap and full CL trajectory are still required", "scientific_conclusion_allowed": False}
    result = {"metadata": metadata, "runs": all_runs, "analysis": analysis}
    (args.output_root / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "seeds": args.seeds, "points": len(pairs_for_corr), "pearson": analysis["effective_rank_to_fresh_gap_pearson"]}, sort_keys=True))


if __name__ == "__main__":
    main()
