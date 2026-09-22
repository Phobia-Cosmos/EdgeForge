#!/usr/bin/env python3
"""Incrementally analyze every ISRUC sequence and epoch on CPU.

This runner is deliberately resumable.  It writes one feature matrix and one
profile per subject, then builds the held-out identity report only after all
subjects are complete.  Raw EEG is read with mmap and is never modified.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_epoch_features():
    path = Path(__file__).with_name("analyze-eeg-subject-fingerprints.py")
    spec = importlib.util.spec_from_file_location("edgeforge_fingerprint_features", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load feature extractor: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.epoch_features


def feature_names(channels: int = 8) -> list[str]:
    names = []
    names.extend(f"log_rms_ch{c}" for c in range(channels))
    names.extend(f"log_rms_centered_ch{c}" for c in range(channels))
    names.extend(f"roughness_ch{c}" for c in range(channels))
    names.extend(f"diff_ratio_ch{c}" for c in range(channels))
    for band in ("delta", "theta", "alpha", "beta", "gamma"):
        names.extend(f"relative_{band}_power_ch{c}" for c in range(channels))
    names.extend(f"spectral_entropy_ch{c}" for c in range(channels))
    names.extend(["signed_connectivity_mean", "signed_connectivity_std", "signed_connectivity_abs_mean"])
    return names


def subject_dirs(root: Path) -> list[Path]:
    return sorted((p for p in root.iterdir() if p.is_dir() and p.name.isdigit() and (p / "data").is_dir()), key=lambda p: int(p.name))


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isruc-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    out = args.output_root.resolve()
    subject_out = out / "subjects"
    subject_out.mkdir(parents=True, exist_ok=True)
    epoch_features = load_epoch_features()
    dirs = subject_dirs(args.isruc_root.resolve())
    if args.max_subjects > 0:
        dirs = dirs[: args.max_subjects]
    manifest_path = out / "progress.json"
    progress = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"schema_version": 1, "dataset": "ISRUC", "subjects": {}, "scientific_conclusion_allowed": False}
    names = feature_names()
    started = time.time()
    for subject_dir in dirs:
        sid = int(subject_dir.name)
        key = str(sid)
        profile_path = subject_out / f"subject-{sid:03d}.json"
        matrix_path = subject_out / f"subject-{sid:03d}-features.npy"
        if profile_path.exists() and matrix_path.exists() and progress.get("subjects", {}).get(key, {}).get("complete"):
            print(json.dumps({"event": "skip_subject", "subject": sid}), flush=True)
            continue
        files = sorted(subject_dir.joinpath("data").glob("*.npy"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
        rows: list[np.ndarray] = []
        sequence_profiles: list[dict[str, Any]] = []
        labels_total: dict[str, int] = {}
        for file_index, path in enumerate(files, start=1):
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            label_path = subject_dir / "label" / path.name
            labels = np.load(label_path, mmap_mode="r", allow_pickle=False).reshape(-1) if label_path.exists() else np.full(values.shape[0], -1)
            if values.ndim != 3:
                raise ValueError(f"invalid ISRUC array {path}: {values.shape}")
            count = min(values.shape[0], labels.shape[0])
            seq_rows = [epoch_features(values[i], 100.0) for i in range(count)]
            seq = np.asarray(seq_rows, dtype=np.float32)
            rows.extend(seq)
            unique, counts = np.unique(np.asarray(labels[:count]), return_counts=True)
            label_counts = {str(int(k)): int(v) for k, v in zip(unique, counts)}
            for label, count_label in label_counts.items():
                labels_total[label] = labels_total.get(label, 0) + count_label
            sequence_profiles.append({"sequence": path.stem, "file": str(path), "n_epochs": int(count), "labels": label_counts, "median": np.median(seq, axis=0).tolist(), "mad": np.median(np.abs(seq - np.median(seq, axis=0)), axis=0).tolist(), "mean": np.mean(seq, axis=0).tolist(), "std": np.std(seq, axis=0).tolist()})
            progress.setdefault("subjects", {})[key] = {"status": "processing", "files_done": file_index, "files_total": len(files), "epochs_done": len(rows)}
            atomic_json(manifest_path, progress)
            print(json.dumps({"event": "sequence_done", "subject": sid, "sequence": path.stem, "sequence_index": file_index, "sequence_total": len(files), "epochs": count}, ensure_ascii=False), flush=True)
        matrix = np.asarray(rows, dtype=np.float32)
        tmp_matrix = matrix_path.with_suffix(".npy.tmp")
        with tmp_matrix.open("wb") as handle:
            np.save(handle, matrix)
        tmp_matrix.replace(matrix_path)
        median = np.median(matrix, axis=0)
        mad = np.median(np.abs(matrix - median), axis=0)
        profile = {"schema_version": 1, "dataset": "ISRUC", "subject": sid, "n_sequences": len(files), "n_epochs": int(len(matrix)), "feature_names": names, "median": median.tolist(), "mad": mad.tolist(), "mean": np.mean(matrix, axis=0).tolist(), "std": np.std(matrix, axis=0).tolist(), "label_counts": labels_total, "sequences": sequence_profiles, "scientific_conclusion_allowed": False}
        atomic_json(profile_path, profile)
        progress.setdefault("subjects", {})[key] = {"complete": True, "n_sequences": len(files), "n_epochs": int(len(matrix))}
        atomic_json(manifest_path, progress)
        print(json.dumps({"event": "subject_complete", "subject": sid, "sequences": len(files), "epochs": len(matrix), "elapsed_seconds": round(time.time() - started, 1)}), flush=True)

    complete_ids = [int(k) for k, v in progress.get("subjects", {}).items() if v.get("complete")]
    if len(complete_ids) != len(dirs):
        print(json.dumps({"status": "partial", "completed_subjects": len(complete_ids), "total_subjects": len(dirs)}), flush=True)
        return
    all_features: list[np.ndarray] = []
    all_subjects: list[np.ndarray] = []
    all_groups: list[str] = []
    profiles: dict[str, Any] = {}
    for sid in complete_ids:
        matrix = np.load(subject_out / f"subject-{sid:03d}-features.npy", mmap_mode="r")
        profile = json.loads((subject_out / f"subject-{sid:03d}.json").read_text())
        profiles[str(sid)] = {k: profile[k] for k in ("n_sequences", "n_epochs", "feature_names", "median", "mad", "mean", "std", "label_counts")}
        all_features.append(np.asarray(matrix))
        all_subjects.append(np.full(len(matrix), sid, dtype=np.int64))
        sequence_rows = []
        for seq in profile["sequences"]:
            sequence_rows.extend([f"{sid}:{seq['sequence']}"] * int(seq["n_epochs"]))
        all_groups.extend(sequence_rows)
    features = np.concatenate(all_features, axis=0)
    subjects = np.concatenate(all_subjects, axis=0)
    groups = np.asarray(all_groups, dtype=object)
    train = np.zeros(len(subjects), dtype=bool)
    test = np.zeros(len(subjects), dtype=bool)
    for sid_value in np.unique(subjects):
        sid = int(sid_value)
        indices = np.flatnonzero(subjects == sid_value)
        unique_groups = list(dict.fromkeys(str(groups[i]) for i in indices))
        cutoff = max(1, len(unique_groups) // 2)
        train_groups = set(unique_groups[:cutoff])
        train[indices] = np.asarray([str(groups[i]) in train_groups for i in indices])
        test[indices] = ~train[indices]
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=600, solver="lbfgs", multi_class="auto"))
    probe.fit(features[train], subjects[train])
    prediction = probe.predict(features[test])
    centroids = {int(sid): np.median(features[train & (subjects == sid)], axis=0) for sid in np.unique(subjects)}
    labels = sorted(centroids)
    centroid_matrix = np.asarray([centroids[sid] for sid in labels])
    centroid_prediction = np.asarray([labels[i] for i in ((features[test, None, :] - centroid_matrix[None, :, :]) ** 2).mean(axis=-1).argmin(axis=1)])
    within = []
    for sid in labels:
        values = features[subjects == sid]
        within.append(float(np.median(np.linalg.norm(values - np.median(values, axis=0), axis=1))))
    between = []
    for i, left in enumerate(labels):
        for right in labels[i + 1 :]:
            between.append(float(np.linalg.norm(centroids[left] - centroids[right])))
    summary = {"schema_version": 1, "dataset": "ISRUC", "sample_count": int(len(features)), "subject_count": int(len(labels)), "sequence_count": int(len(groups) and len(set(groups))), "feature_dim": int(features.shape[1]), "split": "all sequences per subject; first half sequences train, second half held out", "nearest_centroid": {"accuracy": float(accuracy_score(subjects[test], centroid_prediction)), "balanced_accuracy": float(balanced_accuracy_score(subjects[test], centroid_prediction))}, "logistic_subject_probe": {"accuracy": float(accuracy_score(subjects[test], prediction)), "balanced_accuracy": float(balanced_accuracy_score(subjects[test], prediction))}, "within_subject_profile_radius_mean": float(np.mean(within)), "between_subject_centroid_distance_median": float(np.median(between)), "between_within_ratio": float(np.median(between) / max(np.mean(within), 1e-12)), "subject_profiles": profiles, "scientific_conclusion_allowed": False}
    atomic_json(out / "complete-summary.json", summary)
    (out / "all-features.npy").unlink(missing_ok=True)
    np.save(out / "all-subjects.npy", subjects)
    projection = PCA(n_components=2, random_state=0).fit_transform(StandardScaler().fit_transform(features))
    fig, ax = plt.subplots(figsize=(11, 8))
    scatter = ax.scatter(projection[:, 0], projection[:, 1], c=subjects, cmap="turbo", s=4, alpha=0.35, rasterized=True)
    ax.set_title("ISRUC complete subject fingerprint projection\nall subjects, all sequences, all epochs; descriptive")
    ax.set_xlabel("PCA-1 of EEG fingerprint features")
    ax.set_ylabel("PCA-2 of EEG fingerprint features")
    ax.grid(alpha=0.2)
    fig.colorbar(scatter, ax=ax, label="subject id")
    fig.savefig(out / "complete-subject-fingerprint-pca.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    sequence_distances = []
    for sid in labels:
        p = json.loads((subject_out / f"subject-{sid:03d}.json").read_text())
        center = np.asarray(p["median"])
        distances = [float(np.linalg.norm(np.asarray(seq["median"]) - center)) for seq in p["sequences"]]
        sequence_distances.append({"subject": int(sid), "n_sequences": len(distances), "mean_distance": float(np.mean(distances)), "median_distance": float(np.median(distances)), "max_distance": float(np.max(distances)), "distances": distances})
    atomic_json(out / "sequence-drift-by-subject.json", {"dataset": "ISRUC", "subjects": sequence_distances, "scientific_conclusion_allowed": False})
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot([item["distances"] for item in sequence_distances], positions=[item["subject"] for item in sequence_distances], widths=0.6, showfliers=False)
    ax.set_title("ISRUC sequence-to-subject-profile drift")
    ax.set_xlabel("subject id")
    ax.set_ylabel("Euclidean distance in fingerprint feature space")
    ax.grid(alpha=0.2)
    fig.savefig(out / "sequence-drift-by-subject.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    report = ["# ISRUC complete CPU fingerprint analysis", "", f"All {len(labels)} subjects, {summary['sequence_count']} sequences and {len(features)} epochs were processed. Each subject has a complete profile and per-sequence median/MAD in `subjects/`.", "", f"Nearest-centroid accuracy: {summary['nearest_centroid']['accuracy']:.4f}; logistic identity probe accuracy: {summary['logistic_subject_probe']['accuracy']:.4f}; between/within profile distance ratio: {summary['between_within_ratio']:.4f}.", "", "This is a descriptive identity-separability analysis, not an LoP claim. The profiles are constraints for later subject-preserving transformations.", ""]
    (out / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    atomic_json(out / "progress.json", {**progress, "status": "complete", "sample_count": int(len(features)), "sequence_count": summary["sequence_count"], "elapsed_seconds": round(time.time() - started, 1)})
    print(json.dumps({"status": "ok", "subjects": len(labels), "sequences": summary["sequence_count"], "epochs": len(features), "output_root": str(out), "scientific_conclusion_allowed": False}), flush=True)


if __name__ == "__main__":
    main()
