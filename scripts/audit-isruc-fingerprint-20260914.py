#!/usr/bin/env python3
"""Audit the complete ISRUC subject-fingerprint experiment.

This is a read-only audit of the already generated per-epoch feature matrices.
It recomputes the PCA convention used by the report and evaluates identity
probes after removing each handcrafted feature group.  It intentionally does
not instantiate an EdgeForge CNN: the source experiment is waveform -> fixed
feature vector, not waveform -> learned encoder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


GROUPS = {
    "log_rms": slice(0, 8),
    "log_rms_centered": slice(8, 16),
    "roughness": slice(16, 24),
    "diff_ratio": slice(24, 32),
    "relative_band_power": slice(32, 72),
    "spectral_entropy": slice(72, 80),
    "signed_connectivity": slice(80, 83),
}


def load_matrix(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rows: list[np.ndarray] = []
    subjects: list[np.ndarray] = []
    groups: list[str] = []
    names: list[str] | None = None
    for profile_path in sorted((root / "subjects").glob("subject-*.json")):
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        sid = int(profile["subject"])
        matrix = np.load(profile_path.with_name(profile_path.stem + "-features.npy"), mmap_mode="r")
        rows.append(np.asarray(matrix, dtype=np.float32))
        subjects.append(np.full(len(matrix), sid, dtype=np.int64))
        for sequence in profile["sequences"]:
            groups.extend([f"{sid}:{sequence['sequence']}"] * int(sequence["n_epochs"]))
        names = list(profile["feature_names"])
    if not rows or names is None:
        raise RuntimeError(f"no subject matrices found under {root}")
    features = np.concatenate(rows, axis=0)
    subject_ids = np.concatenate(subjects, axis=0)
    sequence_ids = np.asarray(groups, dtype=object)
    if len(sequence_ids) != len(features):
        raise RuntimeError(f"sequence metadata mismatch: {len(sequence_ids)} vs {len(features)}")
    return features, subject_ids, sequence_ids, names


def split_by_sequence(subjects: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.zeros(len(subjects), dtype=bool)
    test = np.zeros(len(subjects), dtype=bool)
    for sid in np.unique(subjects):
        indices = np.flatnonzero(subjects == sid)
        sequence_order = list(dict.fromkeys(str(groups[i]) for i in indices))
        cutoff = max(1, len(sequence_order) // 2)
        train_groups = set(sequence_order[:cutoff])
        train[indices] = np.asarray([str(groups[i]) in train_groups for i in indices])
        test[indices] = ~train[indices]
    return train, test


def probe(features: np.ndarray, subjects: np.ndarray, train: np.ndarray, test: np.ndarray) -> dict[str, float]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=600, solver="lbfgs"),
    )
    model.fit(features[train], subjects[train])
    prediction = model.predict(features[test])
    return {
        "accuracy": float(accuracy_score(subjects[test], prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(subjects[test], prediction)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    features, subjects, sequence_ids, names = load_matrix(root)
    train, test = split_by_sequence(subjects, sequence_ids)
    standardized = StandardScaler().fit_transform(features)
    pca = PCA(n_components=min(20, standardized.shape[1]), random_state=0)
    projection = pca.fit_transform(standardized)
    loadings = pca.components_[:2]
    top_loadings = {}
    for axis, vector in enumerate(loadings, start=1):
        order = np.argsort(np.abs(vector))[::-1][:12]
        top_loadings[f"PCA-{axis}"] = [
            {"feature": names[int(i)], "loading": float(vector[int(i)])} for i in order
        ]

    results = {}
    full = probe(features, subjects, train, test)
    results["all_83_features"] = full
    for group_name, group_slice in GROUPS.items():
        keep = np.ones(features.shape[1], dtype=bool)
        keep[group_slice] = False
        results[f"without_{group_name}"] = {
            "removed_dimensions": int(np.sum(~keep)),
            **probe(features[:, keep], subjects, train, test),
        }

    # Subject-centroid geometry is computed after global standardization, so
    # distances are comparable across feature groups and are not dominated by
    # the raw microvolt scale.
    centroids = np.asarray([np.median(standardized[subjects == sid], axis=0) for sid in np.unique(subjects)])
    within = []
    for sid, centroid in zip(np.unique(subjects), centroids):
        within.append(float(np.median(np.linalg.norm(standardized[subjects == sid] - centroid, axis=1))))
    between = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            between.append(float(np.linalg.norm(centroids[i] - centroids[j])))

    summary = {
        "schema_version": 1,
        "dataset": "ISRUC",
        "source_root": str(root),
        "sample_count_epochs": int(len(features)),
        "subject_count": int(len(np.unique(subjects))),
        "sequence_count": int(len(set(sequence_ids.tolist()))),
        "feature_dim": int(features.shape[1]),
        "feature_names": names,
        "feature_groups": {name: {"start": sl.start, "stop": sl.stop, "dimensions": sl.stop - sl.start} for name, sl in GROUPS.items()},
        "split": "within-subject sequence-disjoint; first half sequences train, second half held out",
        "pca": {
            "standardization": "column-wise StandardScaler over all epoch rows",
            "fit_rows": int(len(features)),
            "explained_variance_ratio_first_10": pca.explained_variance_ratio_[:10].tolist(),
            "explained_variance_ratio_first_2_sum": float(np.sum(pca.explained_variance_ratio_[:2])),
            "top_absolute_loadings": top_loadings,
        },
        "identity_probe": results,
        "standardized_geometry": {
            "within_subject_median_radius_mean": float(np.mean(within)),
            "between_subject_centroid_distance_median": float(np.median(between)),
            "between_within_ratio": float(np.median(between) / max(np.mean(within), 1e-12)),
        },
        "interpretation": {
            "cnn_encoder_used": False,
            "labels_used_as_features": False,
            "pca_points_are_epochs": True,
            "scientific_conclusion_allowed": False,
        },
    }
    (output / "audit-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # A compact audit plot: PCA variance and leave-one-group-out probe accuracy.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(np.arange(1, 11), np.cumsum(pca.explained_variance_ratio_[:10]), marker="o")
    axes[0].set(xlabel="number of PCA components", ylabel="cumulative explained variance", title="Standardized 83-D fingerprint PCA")
    axes[0].set_ylim(0, 1)
    labels = list(results)
    values = [results[label]["accuracy"] for label in labels]
    axes[1].bar(np.arange(len(labels)), values, color=["tab:blue"] + ["tab:orange"] * (len(labels) - 1))
    axes[1].axhline(1.0 / len(np.unique(subjects)), color="black", linestyle="--", linewidth=1, label="98-class chance")
    axes[1].set_xticks(np.arange(len(labels)), [label.replace("without_", "−") for label in labels], rotation=45, ha="right")
    axes[1].set(ylabel="held-out identity accuracy", title="Sequence-disjoint logistic probe")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "fingerprint-audit.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = [
        "# ISRUC fingerprint audit (2026-09-14)",
        "",
        f"The audit re-read {len(features):,} epoch rows from {len(np.unique(subjects))} subjects and {len(set(sequence_ids.tolist())):,} sequences.",
        "",
        f"PCA is fit to column-standardized handcrafted features; the first two components explain {summary['pca']['explained_variance_ratio_first_2_sum']:.4f} of variance.",
        "",
        f"All-feature held-out logistic identity accuracy is {full['accuracy']:.4f}; the split holds out complete sequences within each subject.",
        "",
        "The ablation table in `audit-summary.json` shows which feature groups carry identity information. This is a data-level fingerprint audit, not a CNN embedding or an LoP test.",
        "",
        "PCA axis signs are arbitrary: multiplying one component by -1 gives the same projection. Distances, explained variance and clustering are the interpretable quantities.",
    ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(output), "epochs": len(features), "subjects": len(np.unique(subjects)), "sequences": len(set(sequence_ids.tolist())), "pca12": summary["pca"]["explained_variance_ratio_first_2_sum"], "identity_accuracy": full["accuracy"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
