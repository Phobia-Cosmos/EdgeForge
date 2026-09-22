#!/usr/bin/env python3
"""Closed-set EEG subject identity probes for raw and encoded representations.

This experiment treats subject id as the only prediction target.  It never
loads the sleep/emotion labels.  A complete sequence/trial is one observation:
the first half of each subject's groups is the enrollment split and the second
half is held out.  BrainUICL taps are extracted from a read-only checkpoint;
the external BrainUICL checkout and all source arrays remain untouched.

The protocol answers a closed-set question: can a new sequence from a subject
seen during enrollment be assigned to that subject?  An unseen subject cannot
be assigned a meaningful known id without enrollment; the report therefore
also records the chance level and states this open-set limitation explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, top_k_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STAGES = (
    "raw_waveform",
    "raw_waveform_gain_normalized",
    "eeg_branch",
    "eog_branch",
    "fusion",
    "transformer_layer1",
    "transformer_layer2",
    "transformer_layer3",
    "classifier_input",
    "logits",
)


def _records(root: Path, dataset: str) -> list[tuple[int, str, Path]]:
    rows: list[tuple[int, str, Path]] = []
    if dataset == "ISRUC":
        subjects = sorted((p for p in root.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
        for subject_root in subjects:
            for path in sorted((subject_root / "data").glob("*.npy"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
                rows.append((int(subject_root.name), f"{subject_root.name}:{path.stem}", path))
    elif dataset == "FACED":
        subjects = sorted((p for p in root.iterdir() if p.is_dir() and p.name.startswith("sub-")), key=lambda p: int(p.name.split("-")[-1]))
        for subject_root in subjects:
            for path in sorted((subject_root / "data").glob("*.npy"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
                rows.append((int(subject_root.name.split("-")[-1]), f"{subject_root.name}:{path.stem}", path))
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    if not rows:
        raise RuntimeError(f"no processed sequences found under {root}")
    return rows


def _subject_split(rows: list[tuple[int, str, Path]], seed: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Random but deterministic group split; all observations remain sequence-level."""

    train = np.zeros(len(rows), dtype=bool)
    test = np.zeros(len(rows), dtype=bool)
    manifest: list[dict[str, Any]] = []
    by_subject: dict[int, list[int]] = {}
    for index, (subject, _group, _path) in enumerate(rows):
        by_subject.setdefault(subject, []).append(index)
    for subject, indices in sorted(by_subject.items()):
        order = np.asarray(indices, dtype=np.int64)
        np.random.default_rng(seed + int(subject) * 1009).shuffle(order)
        cutoff = max(1, min(len(order) - 1, len(order) // 2))
        train_indices = order[:cutoff]
        test_indices = order[cutoff:]
        train[train_indices] = True
        test[test_indices] = True
        manifest.append({
            "subject": int(subject),
            "train_groups": [rows[int(i)][1] for i in train_indices],
            "test_groups": [rows[int(i)][1] for i in test_indices],
        })
    return train, test, manifest


def _downsample(values: np.ndarray, points: int = 128) -> np.ndarray:
    """Mean-pool a [T,C,S] sequence to [C,points] without labels."""

    epochs, channels, samples = values.shape
    edges = np.linspace(0, samples, points + 1, dtype=np.int64)
    output = np.empty((epochs, channels, points), dtype=np.float32)
    for index in range(points):
        left, right = int(edges[index]), int(edges[index + 1])
        right = max(left + 1, right)
        output[:, :, index] = values[:, :, left:right].mean(axis=-1)
    return np.median(output, axis=0).reshape(-1)


def _raw_representations(values: np.ndarray) -> dict[str, np.ndarray]:
    raw = np.asarray(values, dtype=np.float32)
    gain = np.sqrt(np.mean(np.square(raw), axis=(1, 2), keepdims=True)).astype(np.float32)
    gain = np.maximum(gain, 1e-12)
    return {
        "raw_waveform": _downsample(raw),
        "raw_waveform_gain_normalized": _downsample(raw / gain),
    }


def _load_blocks(brainuicl_root: Path, checkpoint_root: Path, dataset: str, seed: int, device):
    sys.path.insert(0, str(brainuicl_root.resolve()))
    import torch
    from model.pretrain_net import FeatureExtractor, FeatureExtractorFACED, SleepMLP, TransformerEncoder

    args = SimpleNamespace(dataset=dataset, device=device)
    frontend = FeatureExtractorFACED(args) if dataset == "FACED" else FeatureExtractor(args)
    blocks = [frontend, TransformerEncoder(args), SleepMLP(args)]
    names = ("feature_extractor", "feature_encoder", "sleep_classifier")
    for block, name in zip(blocks, names):
        path = checkpoint_root / f"{name}_parameter_{seed}.pkl"
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            state = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(path, map_location=device)
        block.load_state_dict(state)
        block.eval().to(device)
    return blocks


def _encoded_representations(blocks, values, dataset: str, torch):
    channels = 8 if dataset == "ISRUC" else 32
    samples = 3000 if dataset == "ISRUC" else 2500
    batch = values.shape[0]
    flat = values.reshape(batch * 20, channels, samples)
    if dataset == "ISRUC":
        eeg = flat[:, 2:]
        eog = flat[:, :2]
        eeg_conv = blocks[0].FEBlock_EEG(eeg)
        eog_conv = blocks[0].FEBlock_EOG(eog)
        eeg_pooled = blocks[0].avg(eeg_conv).reshape(batch, 20, 512)
        eog_pooled = blocks[0].avg(eog_conv).reshape(batch, 20, 512)
        fused = blocks[0].fusion(torch.cat((eeg_pooled.reshape(batch * 20, 1, 512), eog_pooled.reshape(batch * 20, 1, 512)), dim=2)).reshape(batch, 20, 512)
    else:
        eog = flat[:, :1]
        eeg = flat[:, 1:]
        conv = blocks[0].features(torch.cat((eog, eeg), dim=1))
        fused = blocks[0].avg(conv).reshape(batch, 20, 512)
        eeg_pooled = fused
        eog_pooled = fused
    encoded = fused
    stage_values: dict[str, torch.Tensor] = {
        "eeg_branch": eeg_pooled,
        "eog_branch": eog_pooled,
        "fusion": fused,
    }
    # BrainUICL declares three repetitions of the same EncoderLayer.  Calling
    # the layer explicitly exposes the output after each repetition rather
    # than collapsing them into one final Transformer tensor.
    for index in range(1, 4):
        encoded = blocks[1].encoder.encoder(encoded)
        stage_values[f"transformer_layer{index}"] = encoded
    classifier_input = blocks[2].sleep_stage_mlp(encoded)
    logits = blocks[2].sleep_stage_classifier(classifier_input).permute(0, 2, 1).contiguous()
    stage_values["classifier_input"] = classifier_input
    stage_values["logits"] = logits
    # Median over the 20 epoch tokens gives one identity observation per
    # sequence and prevents adjacent epochs from leaking into the split.
    return {name: tensor.detach().float().median(dim=1).values.cpu().numpy() for name, tensor in stage_values.items()}


def _probe(matrix: np.ndarray, subjects: np.ndarray, train: np.ndarray, test: np.ndarray, probe_dim: int) -> tuple[dict[str, Any], np.ndarray]:
    if not train.any() or not test.any():
        raise RuntimeError("subject split produced an empty train or test partition; select at least two groups per subject")
    labels = np.asarray(sorted(np.unique(subjects[train])), dtype=np.int64)
    # A compact randomized PCA bottleneck keeps the multinomial probe tractable
    # on the full ISRUC set while preserving a fixed, linear comparison across
    # all taps.  The reported feature_dim remains the original representation
    # width; probe_dim records the bottleneck used by the classifier.
    steps = [StandardScaler()]
    if probe_dim > 0 and matrix.shape[1] > probe_dim:
        steps.append(PCA(n_components=min(probe_dim, matrix.shape[1], max(1, int(train.sum()) - 1)), svd_solver="randomized", random_state=0))
    steps.append(LogisticRegression(max_iter=350, solver="lbfgs"))
    model = make_pipeline(*steps)
    model.fit(matrix[train], subjects[train])
    prediction = model.predict(matrix[test])
    probabilities = model.predict_proba(matrix[test])
    result = {
        "accuracy_top1": float(accuracy_score(subjects[test], prediction)),
        "balanced_accuracy_top1": float(balanced_accuracy_score(subjects[test], prediction)),
        "top5_accuracy": float(top_k_accuracy_score(subjects[test], probabilities, k=min(5, len(labels)), labels=labels)),
        "chance_top1": float(1.0 / len(labels)),
        "known_subject_count": int(len(labels)),
        "train_sequences": int(train.sum()),
        "test_sequences": int(test.sum()),
        "feature_dim": int(matrix.shape[1]),
        "probe_dim": int(probe_dim),
        "confusion_matrix": confusion_matrix(subjects[test], prediction, labels=labels).tolist(),
        "labels": labels.tolist(),
    }
    return result, prediction


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    root = args.data_root.resolve()
    rows = _records(root, args.dataset)
    if args.max_sequences > 0 and args.max_sequences < len(rows):
        # Smoke runs must retain more than one identity.  Select complete
        # groups round-robin over subjects instead of taking the lexicographic
        # prefix (which can contain only subject 1 for ISRUC).
        by_subject: dict[int, list[tuple[int, str, Path]]] = {}
        for row in rows:
            by_subject.setdefault(row[0], []).append(row)
        selected: list[tuple[int, str, Path]] = []
        positions = {subject: 0 for subject in by_subject}
        subjects_order = [subject for subject in sorted(by_subject) if len(by_subject[subject]) >= 2]
        # Keep two groups per selected subject so even a tiny smoke run has a
        # non-empty enrollment and held-out split.
        for subject in subjects_order:
            if len(selected) + 2 > args.max_sequences:
                break
            selected.extend(by_subject[subject][:2])
            positions[subject] = 2
        while len(selected) < args.max_sequences:
            advanced = False
            for subject in subjects_order:
                position = positions[subject]
                if position < len(by_subject[subject]) and len(selected) < args.max_sequences:
                    selected.append(by_subject[subject][position])
                    positions[subject] = position + 1
                    advanced = True
            if not advanced:
                break
        rows = selected
    train, test, split_manifest = _subject_split(rows, args.seed)
    if len(np.unique(np.asarray([row[0] for row in rows]))) < 2:
        raise RuntimeError("identity probe needs at least two subjects")
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    blocks = _load_blocks(args.brainuicl_root, args.checkpoint_root, args.dataset, args.checkpoint_seed, device)
    collected: dict[str, list[np.ndarray]] = {name: [] for name in STAGES}
    subjects: list[int] = []
    groups: list[str] = []
    batch_values: list[np.ndarray] = []
    batch_meta: list[tuple[int, str]] = []
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    def flush_batch() -> None:
        if not batch_values:
            return
        values = torch.from_numpy(np.stack(batch_values)).to(device)
        with torch.inference_mode():
            encoded = _encoded_representations(blocks, values, args.dataset, torch)
        for name, vectors in encoded.items():
            collected[name].extend(np.asarray(vectors, dtype=np.float32))
        batch_values.clear()
        batch_meta.clear()

    for index, (subject, group, path) in enumerate(rows, start=1):
        values = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
        expected = (20, 8, 3000) if args.dataset == "ISRUC" else (20, 32, 2500)
        if tuple(values.shape) != expected:
            raise ValueError(f"expected {path} to have shape {expected}, got {values.shape}")
        raw = _raw_representations(values)
        collected["raw_waveform"].append(raw["raw_waveform"])
        collected["raw_waveform_gain_normalized"].append(raw["raw_waveform_gain_normalized"])
        batch_values.append(values)
        batch_meta.append((subject, group))
        if len(batch_values) >= args.batch_size:
            flush_batch()
        subjects.append(subject)
        groups.append(group)
        if index % 100 == 0 or index == len(rows):
            print(json.dumps({"event": "sequence_done", "index": index, "total": len(rows), "dataset": args.dataset}), flush=True)
    flush_batch()

    y = np.asarray(subjects, dtype=np.int64)
    g = np.asarray(groups, dtype=object)
    # The frontend loop appended raw vectors before the batch loop, while each
    # encoded stage appended one vector per flushed sequence in input order.
    for name in STAGES:
        matrix = np.asarray(collected[name], dtype=np.float32)
        if len(matrix) != len(rows):
            raise RuntimeError(f"stage {name} has {len(matrix)} rows, expected {len(rows)}")
        if args.save_vectors:
            np.save(output / f"{name}-sequence-vectors.npy", matrix)

    stage_results: dict[str, Any] = {}
    prediction_cache: dict[str, np.ndarray] = {}
    for name in STAGES:
        matrix = np.asarray(collected[name], dtype=np.float32)
        stage_results[name], prediction_cache[name] = _probe(matrix, y, train, test, args.probe_dim)

    # Compact visualizations: accuracy by tap and PCA of the held-out identity
    # geometry.  These figures are descriptive; the numeric probe is primary.
    names = list(STAGES)
    accuracy = [stage_results[name]["accuracy_top1"] for name in names]
    chance = 1.0 / len(np.unique(y))
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.bar(np.arange(len(names)), accuracy, color="#3568a8")
    ax.axhline(chance, color="black", linestyle="--", linewidth=1, label=f"chance = {chance:.4f}")
    ax.set_xticks(np.arange(len(names)), names, rotation=40, ha="right")
    ax.set_ylabel("held-out subject-id top-1 accuracy")
    ax.set_title(f"{args.dataset}: closed-set subject identification by representation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "subject-id-accuracy-by-stage.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    # PCA figures are descriptive only; cap the plotted sample so visualization
    # does not dominate a full-dataset run.
    plot_rng = np.random.default_rng(args.seed)
    plot_indices = np.arange(len(y)) if len(y) <= args.plot_max_sequences else np.sort(plot_rng.choice(len(y), size=args.plot_max_sequences, replace=False))
    for name in names:
        matrix = np.asarray(collected[name], dtype=np.float32)
        projection = PCA(n_components=2, random_state=0).fit_transform(StandardScaler().fit_transform(matrix[plot_indices]))
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(projection[:, 0], projection[:, 1], c=y[plot_indices], cmap="turbo", s=9, alpha=0.6)
        ax.set_title(f"{args.dataset} · {name} · sequence-level PCA")
        ax.set_xlabel("PCA-1")
        ax.set_ylabel("PCA-2")
        fig.tight_layout()
        fig.savefig(output / f"pca-{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    result = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment": "closed-set-subject-identity-probe-v1",
        "dataset": args.dataset,
        "data_root": str(root),
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "checkpoint_seed": int(args.checkpoint_seed),
        "device": str(device),
        "sequence_count": int(len(rows)),
        "subject_count": int(len(np.unique(y))),
        "input_shapes": {"ISRUC": [20, 8, 3000], "FACED": [20, 32, 2500]},
        "target": "subject_id_only; source sleep/emotion labels were not loaded",
        "split": "deterministic random half of complete sequence/trial groups per subject; no epoch-level leakage",
        "split_manifest": split_manifest,
        "open_set_boundary": "A subject absent from enrollment has no known subject-id class. The reported accuracies are closed-set identification scores.",
        "stages": stage_results,
        "scientific_conclusion_allowed": False,
        "artifacts": {"accuracy_plot": "subject-id-accuracy-by-stage.png", "pca_pattern": "pca-<stage>.png", "vectors_saved": bool(args.save_vectors)},
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    (output / "subject-id-summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "created_at": result["created_at"],
        "script": "scripts/subject_identity_probe.py",
        "experiment": result["experiment"],
        "dataset": args.dataset,
        "arguments": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "sequence_count": result["sequence_count"],
        "subject_count": result["subject_count"],
        "device": result["device"],
        "checkpoint_seed": result["checkpoint_seed"],
        "probe_dim": int(args.probe_dim),
        "save_vectors": bool(args.save_vectors),
        "runtime": result["runtime"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        f"# {args.dataset} subject identity probe",
        "",
        f"This closed-set experiment uses {len(rows):,} complete sequences/trials from {len(np.unique(y))} subjects. The target is subject id only; sleep/emotion labels are not loaded.",
        "",
        "Enrollment uses a deterministic random half of complete groups per subject and evaluation uses the other half. A high score means a new group from a known subject can be assigned to that subject; it does not identify an unseen subject without an enrollment profile.",
        "",
        "| representation | top-1 | balanced top-1 | top-5 | chance | feature dim |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in names:
        row = stage_results[name]
        report.append(f"| {name} | {row['accuracy_top1']:.4f} | {row['balanced_accuracy_top1']:.4f} | {row['top5_accuracy']:.4f} | {row['chance_top1']:.4f} | {row['feature_dim']} |")
    report.extend([
        "",
        "The raw waveform stages use sequence-level median mean-pooled samples (128 points per channel), with and without per-epoch gain normalization. The learned stages are sequence-level medians of 20 epoch tokens.",
        "",
        "This is an identity separability audit, not a biometric authentication claim and not a LoP outcome. Repeat with session-disjoint and class-balanced splits before interpreting the representation as subject-invariant biology.",
    ])
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ISRUC", "FACED"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--brainuicl-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint-seed", type=int, default=4321)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--probe-dim", type=int, default=128, help="PCA width used before the logistic subject probe; 0 disables the bottleneck")
    parser.add_argument("--plot-max-sequences", type=int, default=5000)
    parser.add_argument("--save-vectors", action="store_true", help="save per-stage sequence vectors (can be large on ISRUC)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    result = run(args)
    print(json.dumps({"status": "ok", "dataset": result["dataset"], "sequence_count": result["sequence_count"], "subject_count": result["subject_count"], "output": str(args.output.resolve())}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
