#!/usr/bin/env python3
"""Measure subject-invariant information in waveform and BrainUICL features.

This read-only CPU audit works on already extracted feature/embedding matrices.
It uses complete-sequence splits, so adjacent epochs from one sequence cannot
leak into both the identity probe train and test sets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def sequence_medians(matrix: np.ndarray, subjects: np.ndarray, groups: np.ndarray):
    ordered = list(dict.fromkeys(str(value) for value in groups))
    rows, y = [], []
    for group in ordered:
        mask = groups == group
        rows.append(np.median(matrix[mask], axis=0))
        values = np.unique(subjects[mask])
        if len(values) != 1:
            raise ValueError(f"group {group!r} contains multiple subjects")
        y.append(int(values[0]))
    return np.asarray(rows, dtype=np.float64), np.asarray(y, dtype=np.int64), ordered


def split_by_sequence(subjects: np.ndarray, groups: np.ndarray):
    train = np.zeros(len(subjects), dtype=bool)
    test = np.zeros(len(subjects), dtype=bool)
    for sid in np.unique(subjects):
        indices = np.flatnonzero(subjects == sid)
        ordered = list(dict.fromkeys(str(groups[i]) for i in indices))
        cutoff = max(1, len(ordered) // 2)
        train[indices] = np.asarray([str(groups[i]) in set(ordered[:cutoff]) for i in indices])
        test[indices] = ~train[indices]
    return train, test


def probe(matrix: np.ndarray, subjects: np.ndarray, groups: np.ndarray) -> float:
    train, test = split_by_sequence(subjects, groups)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=700, solver="lbfgs"))
    model.fit(matrix[train], subjects[train])
    return float(np.mean(model.predict(matrix[test]) == subjects[test]))


def feature_icc(sequence_matrix: np.ndarray, sequence_subjects: np.ndarray):
    """One-way ICC(1,1) for sequence-level profiles, one value per dimension."""
    values = []
    subjects = np.unique(sequence_subjects)
    for column in range(sequence_matrix.shape[1]):
        x = sequence_matrix[:, column]
        groups = [x[sequence_subjects == sid] for sid in subjects]
        counts = np.asarray([len(group) for group in groups], dtype=np.int64)
        if np.any(counts < 2):
            values.append(np.nan)
            continue
        means = np.asarray([np.mean(group) for group in groups])
        overall = float(np.mean(x))
        between_ss = float(np.sum(counts * (means - overall) ** 2))
        within_ss = float(sum(np.sum((group - mean) ** 2) for group, mean in zip(groups, means)))
        between_ms = between_ss / max(len(subjects) - 1, 1)
        within_ms = within_ss / max(len(x) - len(subjects), 1)
        k = float(np.mean(counts))
        denominator = between_ms + (k - 1.0) * within_ms
        values.append((between_ms - within_ms) / denominator if denominator > 1e-15 else 0.0)
    return np.asarray(values, dtype=np.float64)


def group_for_name(name: str) -> str:
    if name.startswith("log_rms_") and not name.startswith("log_rms_centered"):
        return "absolute_gain"
    if name.startswith("absolute_") or name.startswith("mean_") or name.startswith("median_") or name.startswith("std_"):
        return "scale_sensitive"
    if name.startswith("log_rms_centered") or name.startswith("diff_ratio") or name.startswith("relative_"):
        return "gain_invariant_ratio"
    if name.startswith("signed_corr"):
        return "spatial_connectivity"
    if name.startswith("spectral_") or name.startswith("psd_"):
        return "spectral_shape"
    if name.startswith("roughness") or name.startswith("hjorth") or name.startswith("line_length") or name.startswith("skew") or name.startswith("kurtosis") or name.startswith("crest") or name.startswith("zero_cross"):
        return "morphology"
    return "other"


def summarize_matrix(matrix, subjects, groups, names, label):
    sequence, sequence_subjects, sequence_groups = sequence_medians(matrix, subjects, groups)
    icc = feature_icc(sequence, sequence_subjects)
    order = np.argsort(np.nan_to_num(icc, nan=-1.0))[::-1]
    dimensions = [20, 50, 100, 150, 200, matrix.shape[1]]
    dimensions = sorted(set(min(value, matrix.shape[1]) for value in dimensions))
    probe_by_k = {str(k): probe(sequence[:, order[:k]], sequence_subjects, np.asarray(sequence_groups, dtype=object)) for k in dimensions}
    groups_summary = {}
    for group_name in sorted(set(group_for_name(name) for name in names)):
        mask = np.asarray([group_for_name(name) == group_name for name in names])
        group_values = icc[mask]
        groups_summary[group_name] = {
            "count": int(mask.sum()),
            "median_icc": float(np.nanmedian(group_values)),
            "positive_icc_fraction": float(np.mean(group_values > 0)),
        }
    subset_masks = {
        "all_dimensions": np.ones(len(names), dtype=bool),
        "gain_invariant_no_absolute_scale": np.asarray([group_for_name(name) not in {"absolute_gain", "scale_sensitive"} for name in names]),
        "relative_connectivity_only": np.asarray([group_for_name(name) in {"gain_invariant_ratio", "spatial_connectivity"} for name in names]),
    }
    subset_probes = {}
    for subset_name, mask in subset_masks.items():
        if mask.any():
            subset_probes[subset_name] = {"dimensions": int(mask.sum()), "accuracy": probe(sequence[:, mask], sequence_subjects, np.asarray(sequence_groups, dtype=object))}
    return {
        "label": label,
        "epoch_count": int(len(matrix)),
        "sequence_count": int(len(sequence)),
        "subject_count": int(len(np.unique(subjects))),
        "feature_dim": int(matrix.shape[1]),
        "identity_probe_top_stable_dimensions": probe_by_k,
        "identity_probe_feature_subsets": subset_probes,
        "icc": {"median": float(np.nanmedian(icc)), "positive_fraction": float(np.mean(icc > 0)), "top_features": [{"name": names[int(index)], "icc": float(icc[index])} for index in order[:20]]},
        "feature_groups": groups_summary,
        "sequence_matrix": sequence,
        "sequence_subjects": sequence_subjects,
        "sequence_groups": np.asarray(sequence_groups, dtype=object),
        "icc_values": icc,
    }


def load_brainuicl(directory: Path):
    subjects = np.load(directory / "subjects.npy", allow_pickle=False)
    groups = np.load(directory / "groups.npy", allow_pickle=True).astype(str)
    summary = json.loads((directory / "brainuicl-subject-stability-summary.json").read_text(encoding="utf-8"))
    outputs = []
    for stage, shape in summary["stage_shapes"].items():
        path = directory / f"{stage}-epoch-embeddings.npy"
        if not path.is_file():
            continue
        matrix = np.load(path, mmap_mode="r", allow_pickle=False).astype(np.float64)
        names = [f"{stage}[{index}]" for index in range(matrix.shape[1])]
        outputs.append(summarize_matrix(matrix, subjects, groups, names, stage))
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded-dir", type=Path, required=True)
    parser.add_argument("--brainuicl-dir", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    expanded = args.expanded_dir.resolve()
    matrix = np.load(expanded / "expanded-features.npy", mmap_mode="r", allow_pickle=False).astype(np.float64)
    subjects = np.load(expanded / "subjects.npy", allow_pickle=False)
    groups = np.load(expanded / "groups.npy", allow_pickle=True).astype(str)
    summary = json.loads((expanded / "expanded-fingerprint-summary.json").read_text(encoding="utf-8"))
    names = summary["feature_names"]
    waveform = summarize_matrix(matrix, subjects, groups, names, "expanded_waveform_295d")
    payload = {"schema_version": 1, "waveform": {key: value for key, value in waveform.items() if key not in {"sequence_matrix", "sequence_subjects", "sequence_groups", "icc_values"}}, "brainuicl": []}

    labels, median_iccs, full_probe, top_probe = [], [], [], []
    labels.append("waveform_295d")
    median_iccs.append(waveform["icc"]["median"])
    full_probe.append(waveform["identity_probe_top_stable_dimensions"][str(matrix.shape[1])])
    top_probe.append(waveform["identity_probe_top_stable_dimensions"][str(min(50, matrix.shape[1]))])
    for brain_dir in args.brainuicl_dir:
        audit = load_brainuicl(brain_dir.resolve())
        run_label = brain_dir.name
        run_payload = {"run": run_label, "stages": []}
        for item in audit:
            run_payload["stages"].append({key: value for key, value in item.items() if key not in {"sequence_matrix", "sequence_subjects", "sequence_groups", "icc_values"}})
            labels.append(f"{run_label}:{item['label']}")
            median_iccs.append(item["icc"]["median"])
            full_probe.append(item["identity_probe_top_stable_dimensions"][str(item["feature_dim"])])
            top_probe.append(item["identity_probe_top_stable_dimensions"][str(min(50, item["feature_dim"]))])
        payload["brainuicl"].append(run_payload)

    (output / "subject-invariance-summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    names_plot = [label.replace("brainuicl-subject-stability-", "") for label in labels]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes[0].bar(x, median_iccs, color=["tab:blue" if i == 0 else "tab:orange" for i in range(len(labels))])
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("median sequence-level ICC")
    axes[0].set_title("Feature/tap repeatability")
    axes[1].bar(x - 0.18, full_probe, width=0.36, label="all dimensions")
    axes[1].bar(x + 0.18, top_probe, width=0.36, label="top 50 stable dimensions")
    axes[1].set_ylabel("sequence-disjoint identity accuracy")
    axes[1].set_title("All vs stable dimensions")
    axes[1].legend(fontsize=8)
    axes[2].bar(x, np.asarray(top_probe) - np.asarray(full_probe), color="tab:green")
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_ylabel("top-50 accuracy minus all")
    axes[2].set_title("Effect of selecting stable dimensions")
    axes[2].set_xticks(x, names_plot, rotation=75, ha="right", fontsize=7)
    axes[0].set_xticks(x, names_plot, rotation=75, ha="right", fontsize=7)
    axes[1].set_xticks(x, names_plot, rotation=75, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "subject-invariance-comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    group_names = sorted(waveform["feature_groups"])
    group_values = [waveform["feature_groups"][name]["median_icc"] for name in group_names]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(len(group_names)), group_values, color="tab:purple")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(group_names)), group_names, rotation=35, ha="right")
    ax.set_ylabel("median sequence-level ICC")
    ax.set_title("Waveform feature-family repeatability")
    fig.tight_layout()
    fig.savefig(output / "waveform-feature-family-icc.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    (output / "REPORT.md").write_text("\n".join([
        "# Subject-invariance audit",
        "",
        "This read-only audit aggregates each 20-epoch sequence to one median profile and computes one-way sequence-level ICC per feature or network dimension. Higher ICC means that sequence profiles from the same subject are more reproducible relative to between-subject variation; it is not a biometric authentication score.",
        "",
        "`subject-invariance-comparison.png` compares median ICC and sequence-disjoint identity probes for the expanded waveform bank and each supplied BrainUICL tap. `waveform-feature-family-icc.png` shows which waveform feature families are repeatable.",
        "",
        "The top-k probe orders dimensions by ICC before fitting the classifier. If top-k accuracy is not higher than the all-dimension result, extra dimensions are either useful jointly or not harmful in this pilot; if it is higher, unstable dimensions are likely adding drift/noise. This selection is descriptive and must be confirmed on session-disjoint data.",
        "",
        "BrainUICL stage names refer to EEG/EOG CNN branches, fusion, Transformer outputs, classifier input and logits. A trained checkpoint is required before interpreting a tap as a learned invariant representation; random-init results only measure architectural signal preservation.",
    ]) + "\n", encoding="utf-8")
    (output / "EXPLANATION.md").write_text("\n".join([
        "# 图表说明",
        "",
        "`subject-invariance-comparison.png`：每个标签对应一个波形特征矩阵或 BrainUICL tap。median ICC 越高，说明同一被试不同 sequence 的 profile 越稳定；identity accuracy 是完整 sequence 留出后的身份可分性。两者必须一起看，单独高准确率可能来自睡眠阶段或采集条件混杂。",
        "",
        "`waveform-feature-family-icc.png`：将 295 维按绝对增益、尺度敏感、增益相对比值、空间连接、谱形状和形态学分组，显示每组的中位 sequence-level ICC。绝对功率高不代表真正个体不变，可能只是电极阻抗或放大器增益。",
        "",
        "本审计使用 sequence median，而不是把相邻 epoch 当作独立样本；因此图中的稳定性更接近跨 sequence 复现。ICC 仍不是跨 session 的最终结论，后续需使用明确 session 标签、class-balanced probe 和增益/睡眠阶段校正复核。",
    ]) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(output), "waveform_features": int(matrix.shape[1]), "brainuicl_runs": len(args.brainuicl_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
