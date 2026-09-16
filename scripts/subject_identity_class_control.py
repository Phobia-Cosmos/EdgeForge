#!/usr/bin/env python3
"""Control EEG subject-ID probes for sleep/emotion class composition.

Two complementary controls are reported:

1. A class-histogram-only sequence probe quantifies how much identity can be
   decoded from each subject's task-label proportions alone.
2. Class-conditioned epoch probes hold the sleep/emotion class fixed and use
   the same number of train/test epochs per included subject.  Accuracy above
   chance therefore cannot be explained only by different class proportions.

The BrainUICL checkpoint is read-only.  Subject ID is decoded by a separate
linear probe and never replaces the checkpoint's original task head.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subject_identity_granularity import (  # noqa: E402
    STAGES,
    _encoded_epoch_representations,
    _raw_epoch_representations,
)
from subject_identity_probe import _load_blocks, _records  # noqa: E402


def _group_split(rows: list[tuple[int, str, Path]], seed: int) -> tuple[np.ndarray, np.ndarray]:
    train = np.zeros(len(rows), dtype=bool)
    test = np.zeros(len(rows), dtype=bool)
    by_subject: dict[int, list[int]] = {}
    for index, (subject, _group, _path) in enumerate(rows):
        by_subject.setdefault(subject, []).append(index)
    for subject, values in by_subject.items():
        indices = np.asarray(values, dtype=np.int64)
        np.random.default_rng(seed + subject * 1009).shuffle(indices)
        cutoff = max(1, min(len(indices) - 1, len(indices) // 2))
        train[indices[:cutoff]] = True
        test[indices[cutoff:]] = True
    if not train.any() or not test.any():
        raise RuntimeError("class control requires at least two complete groups per subject")
    return train, test


def _probe(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray, probe_dim: int) -> dict[str, Any]:
    labels = np.asarray(sorted(np.unique(train_y)), dtype=np.int64)
    steps: list[Any] = [StandardScaler()]
    if probe_dim > 0 and train_x.shape[1] > probe_dim:
        steps.append(PCA(n_components=min(probe_dim, train_x.shape[1], max(1, len(train_x) - 1)), svd_solver="randomized", random_state=0))
    # The control needs only a deterministic linear-decodability test. Ridge
    # avoids the iteration-limit sensitivity of 98/123-way multinomial
    # LogisticRegression while preserving the same standardized/PCA feature
    # comparison across representations.
    steps.append(RidgeClassifier(alpha=1.0, class_weight="balanced"))
    model = make_pipeline(*steps)
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)
    return {
        "accuracy_top1": float(accuracy_score(test_y, prediction)),
        "balanced_accuracy_top1": float(balanced_accuracy_score(test_y, prediction)),
        "chance_top1": float(1.0 / len(labels)),
        "subject_count": int(len(labels)),
        "train_samples": int(len(train_y)),
        "test_samples": int(len(test_y)),
        "feature_dim": int(train_x.shape[1]),
        "probe_dim": int(probe_dim),
        "probe": "StandardScaler -> optional randomized PCA -> class-balanced RidgeClassifier(alpha=1.0)",
    }


def _balanced_class_indices(
    labels: np.ndarray,
    subjects: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    class_id: int,
    minimum: int,
    cap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    eligible: list[int] = []
    train_by_subject: dict[int, np.ndarray] = {}
    test_by_subject: dict[int, np.ndarray] = {}
    for subject in np.unique(subjects):
        train_indices = np.flatnonzero(train & (subjects == subject) & (labels == class_id))
        test_indices = np.flatnonzero(test & (subjects == subject) & (labels == class_id))
        if len(train_indices) >= minimum and len(test_indices) >= minimum:
            sid = int(subject)
            eligible.append(sid)
            train_by_subject[sid] = train_indices
            test_by_subject[sid] = test_indices
    if len(eligible) < 2:
        return None
    train_per_subject = min(cap, min(len(train_by_subject[sid]) for sid in eligible))
    test_per_subject = min(cap, min(len(test_by_subject[sid]) for sid in eligible))
    train_selected: list[np.ndarray] = []
    test_selected: list[np.ndarray] = []
    for subject in eligible:
        rng = np.random.default_rng(seed + class_id * 100003 + subject * 9176)
        train_selected.append(rng.choice(train_by_subject[subject], size=train_per_subject, replace=False))
        test_selected.append(rng.choice(test_by_subject[subject], size=test_per_subject, replace=False))
    return (
        np.concatenate(train_selected),
        np.concatenate(test_selected),
        {
            "subjects": eligible,
            "subject_count": len(eligible),
            "train_epochs_per_subject": int(train_per_subject),
            "test_epochs_per_subject": int(test_per_subject),
        },
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    rows = _records(args.data_root.resolve(), args.dataset)
    train_groups, test_groups = _group_split(rows, args.seed)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    blocks = _load_blocks(args.brainuicl_root, args.checkpoint_root, args.dataset, args.checkpoint_seed, device)
    stages = tuple(value.strip() for value in args.stages.split(",") if value.strip())
    invalid = sorted(set(stages) - set(STAGES))
    if invalid:
        raise ValueError(f"unknown stages: {invalid}")

    expected = (20, 8, 3000) if args.dataset == "ISRUC" else (20, 32, 2500)
    class_count = 5 if args.dataset == "ISRUC" else 9
    collected: dict[str, list[np.ndarray]] = {stage: [] for stage in stages}
    group_labels: list[np.ndarray] = []
    group_subjects: list[int] = []
    class_histograms: list[np.ndarray] = []
    for index, (subject, _group, path) in enumerate(rows, start=1):
        values = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
        label_path = path.parent.parent / "label" / path.name
        labels = np.load(label_path, allow_pickle=False).reshape(-1).astype(np.int64, copy=False)
        if tuple(values.shape) != expected or tuple(labels.shape) != (20,):
            raise ValueError(f"unexpected data/label shape for {path}: {values.shape}/{labels.shape}")
        if labels.min() < 0 or labels.max() >= class_count:
            raise ValueError(f"labels outside [0,{class_count - 1}] in {label_path}")
        raw = _raw_epoch_representations(values)
        with torch.inference_mode():
            encoded = _encoded_epoch_representations(blocks, torch.from_numpy(values).to(device), args.dataset, torch)
        for stage in stages:
            collected[stage].append(np.asarray(raw[stage] if stage in raw else encoded[stage], dtype=np.float32))
        group_labels.append(labels)
        group_subjects.append(subject)
        class_histograms.append(np.bincount(labels, minlength=class_count).astype(np.float32) / len(labels))
        if index % 100 == 0 or index == len(rows):
            print(json.dumps({"event": "sequence_done", "index": index, "total": len(rows), "dataset": args.dataset}), flush=True)

    group_subject_array = np.asarray(group_subjects, dtype=np.int64)
    histogram_matrix = np.asarray(class_histograms, dtype=np.float32)
    histogram_probe = _probe(
        histogram_matrix[train_groups],
        group_subject_array[train_groups],
        histogram_matrix[test_groups],
        group_subject_array[test_groups],
        0,
    )

    epoch_labels = np.concatenate(group_labels)
    epoch_subjects = np.repeat(group_subject_array, 20)
    epoch_train = np.repeat(train_groups, 20)
    epoch_test = np.repeat(test_groups, 20)
    matrices = {stage: np.concatenate(values, axis=0) for stage, values in collected.items()}
    class_results: dict[str, Any] = {}
    for class_id in range(class_count):
        selected = _balanced_class_indices(
            epoch_labels,
            epoch_subjects,
            epoch_train,
            epoch_test,
            class_id,
            args.min_epochs_per_subject_class,
            args.max_epochs_per_subject_class,
            args.seed,
        )
        if selected is None:
            class_results[str(class_id)] = {"status": "insufficient_subjects"}
            continue
        train_indices, test_indices, selection = selected
        stage_results = {
            stage: _probe(
                matrix[train_indices],
                epoch_subjects[train_indices],
                matrix[test_indices],
                epoch_subjects[test_indices],
                args.probe_dim,
            )
            for stage, matrix in matrices.items()
        }
        class_results[str(class_id)] = {"status": "ok", "selection": selection, "stages": stage_results}

    stage_macro: dict[str, Any] = {}
    for stage in stages:
        values = [row["stages"][stage]["balanced_accuracy_top1"] for row in class_results.values() if row["status"] == "ok"]
        stage_macro[stage] = {
            "class_count": len(values),
            "macro_balanced_accuracy_mean": float(np.mean(values)) if values else None,
            "macro_balanced_accuracy_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None,
        }

    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    summary = {
        "schema_version": 1,
        "created_at": created_at,
        "experiment": "subject-identity-class-conditioned-control-v1",
        "dataset": args.dataset,
        "data_root": str(args.data_root.resolve()),
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "checkpoint_seed": int(args.checkpoint_seed),
        "split_seed": int(args.seed),
        "device": str(device),
        "sequence_count": len(rows),
        "epoch_count": len(rows) * 20,
        "subject_count": int(len(np.unique(group_subject_array))),
        "class_count": class_count,
        "group_split": "within each subject, half complete groups enrollment and half held-out test",
        "class_histogram_only_sequence_probe": histogram_probe,
        "class_conditioned_epoch_probes": class_results,
        "class_conditioned_macro": stage_macro,
        "control_definition": "Within each task class, retain subjects with sufficient epochs in both group-disjoint splits and sample exactly the same epoch count per subject.",
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__},
        "scientific_conclusion_allowed": False,
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    successful_classes = [key for key, value in class_results.items() if value["status"] == "ok"]
    figure = np.asarray([[class_results[key]["stages"][stage]["balanced_accuracy_top1"] for key in successful_classes] for stage in stages])
    fig, ax = plt.subplots(figsize=(max(8, len(successful_classes) * 1.0), max(4, len(stages) * 0.8)))
    image = ax.imshow(figure, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(successful_classes)), [f"class {key}" for key in successful_classes])
    ax.set_yticks(np.arange(len(stages)), stages)
    ax.set_title(f"{args.dataset}: subject-ID accuracy with task class held fixed")
    for row in range(figure.shape[0]):
        for column in range(figure.shape[1]):
            ax.text(column, row, f"{figure[row, column]:.2f}", ha="center", va="center", color="white" if figure[row, column] < 0.55 else "black", fontsize=8)
    fig.colorbar(image, ax=ax, label="balanced subject-ID top-1")
    fig.tight_layout()
    fig.savefig(output / "class-conditioned-subject-id.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = [
        f"# {args.dataset} subject-ID class-conditioned control",
        "",
        f"This run uses {len(rows):,} complete groups, {len(rows) * 20:,} epochs and {len(np.unique(group_subject_array))} subjects. Subject ID is the probe target; the original task label is used only to construct controls.",
        "",
        f"A classifier that sees only each sequence's {class_count}-D task-label histogram obtains top-1 `{histogram_probe['accuracy_top1']:.4f}` (chance `{histogram_probe['chance_top1']:.4f}`). This estimates identity leakage from class composition alone.",
        "",
        "Class-conditioned probes hold the task class fixed and balance train/test epoch counts across included subjects:",
        "",
        "| representation | macro balanced subject-ID accuracy | class-to-class std |",
        "| --- | ---: | ---: |",
    ]
    for stage in stages:
        row = stage_macro[stage]
        report.append(f"| {stage} | {row['macro_balanced_accuracy_mean']:.4f} | {row['macro_balanced_accuracy_std']:.4f} |")
    report.extend([
        "",
        "Accuracy above each row's recorded chance means subject information remains decodable even after fixing the sleep/emotion class. It still may contain session, electrode, impedance, gain or acquisition-device cues, so this is not proof of immutable biometric identity.",
        "",
        "This control is not an LoP outcome. Its role is to define a cleaner identity signal that can later be tracked over continual-learning checkpoints.",
    ])
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ISRUC", "FACED"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--brainuicl-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint-seed", type=int, default=4321)
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--stages", default="raw_waveform,eeg_branch,transformer_layer1,classifier_input")
    parser.add_argument("--probe-dim", type=int, default=64)
    parser.add_argument("--min-epochs-per-subject-class", type=int, default=5)
    parser.add_argument("--max-epochs-per-subject-class", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    result = run(args)
    print(json.dumps({"status": "ok", "dataset": result["dataset"], "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
