#!/usr/bin/env python3
"""Compare the 83-D EEG fingerprint with a larger waveform feature bank.

This is a CPU, read-only medium-ISRUC experiment.  The expanded bank adds
robust time-domain, Hjorth, absolute spectral, spectral-shape and pairwise
connectivity features.  All identity tests hold out complete sequences within
each subject, so the result measures cross-sequence reproducibility rather
than memorization of adjacent epochs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (13.0, 16.0),
    "beta": (16.0, 30.0),
    "gamma": (30.0, 45.0),
}


def integral(values: np.ndarray, coordinates: np.ndarray) -> float:
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(values, coordinates))


def load_base_features():
    path = Path(__file__).with_name("analyze-eeg-subject-fingerprints.py")
    spec = importlib.util.spec_from_file_location("base_eeg_features", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.epoch_features


def extra_features(epoch: np.ndarray, fs: float) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(epoch, dtype=np.float64)
    channels, samples = values.shape
    names: list[str] = []
    blocks: list[np.ndarray] = []

    def add(label: str, data: np.ndarray) -> None:
        names.extend(f"{label}_ch{c}" for c in range(channels))
        blocks.append(np.asarray(data, dtype=np.float64))

    mean = values.mean(axis=1)
    median = np.median(values, axis=1)
    std = values.std(axis=1)
    rms = np.sqrt(np.mean(values**2, axis=1))
    add("mean", mean)
    add("median", median)
    add("std", std)
    add("skew", skew(values, axis=1, bias=False))
    add("kurtosis", kurtosis(values, axis=1, fisher=True, bias=False))
    add("crest_factor", np.max(np.abs(values), axis=1) / np.maximum(rms, 1e-30))
    add("zero_cross_rate", np.mean(values[:, 1:] * values[:, :-1] < 0, axis=1))
    difference = np.diff(values, axis=1)
    second_difference = np.diff(values, n=2, axis=1)
    activity = np.var(values, axis=1)
    mobility = np.sqrt(np.var(difference, axis=1) / np.maximum(activity, 1e-30))
    complexity = np.sqrt(np.var(second_difference, axis=1) / np.maximum(np.var(difference, axis=1), 1e-30)) / np.maximum(mobility, 1e-30)
    add("hjorth_activity", activity)
    add("hjorth_mobility", mobility)
    add("hjorth_complexity", complexity)
    add("line_length_per_sample", np.mean(np.abs(difference), axis=1))

    absolute_band = np.zeros((channels, len(BANDS)))
    shape = np.zeros((channels, 5))
    shape_names = ["spectral_centroid", "spectral_bandwidth", "spectral_edge95", "spectral_flatness", "spectral_peak_frequency"]
    psd_slope = np.zeros(channels)
    for c, signal in enumerate(values):
        freq, power = welch(signal, fs=fs, nperseg=min(512, samples))
        valid = (freq >= 0.5) & (freq <= min(45.0, fs / 2.0))
        f = freq[valid]
        p = np.maximum(power[valid], 1e-30)
        total = max(integral(p, f), 1e-30)
        for b, (low, high) in enumerate(BANDS.values()):
            mask = (f >= low) & (f < min(high, fs / 2.0))
            absolute_band[c, b] = np.log10(max(integral(p[mask], f[mask]), 1e-30)) if mask.any() else -30.0
        normalized = p / max(float(p.sum()), 1e-30)
        shape[c, 0] = float(np.sum(f * normalized))
        shape[c, 1] = float(np.sqrt(np.sum((f - shape[c, 0]) ** 2 * normalized)))
        cumulative = np.cumsum(normalized)
        shape[c, 2] = float(f[min(np.searchsorted(cumulative, 0.95), len(f) - 1)])
        shape[c, 3] = float(np.exp(np.mean(np.log(p))) / max(float(np.mean(p)), 1e-30))
        shape[c, 4] = float(f[int(np.argmax(p))])
        band = (f >= 1.0) & (f <= 40.0)
        if band.sum() >= 2:
            psd_slope[c] = float(np.polyfit(np.log(f[band]), np.log(p[band]), 1)[0])
    for b, label in enumerate(BANDS):
        add(f"absolute_{label}_log_power", absolute_band[:, b])
    for j, label in enumerate(shape_names):
        add(label, shape[:, j])
    add("psd_log_log_slope", psd_slope)

    corr = np.nan_to_num(np.corrcoef(values), nan=0.0)
    upper = corr[np.triu_indices(channels, k=1)]
    blocks.append(upper)
    names.extend(f"signed_corr_ch{a}_ch{b}" for a in range(channels) for b in range(a + 1, channels))
    return np.concatenate(blocks).astype(np.float32), names


def files(root: Path):
    rows = []
    for split in ("source", "target", "retention"):
        for subject in sorted((p for p in (root / split).iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name)):
            for data in sorted((subject / "data").glob("*.npy"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
                label = subject / "label" / data.name
                if label.is_file():
                    rows.append((int(subject.name), f"{split}:{subject.name}:{data.stem}", data, label))
    return rows


def split(subjects, groups):
    train = np.zeros(len(subjects), dtype=bool)
    test = np.zeros(len(subjects), dtype=bool)
    for sid in np.unique(subjects):
        idx = np.flatnonzero(subjects == sid)
        ordered = list(dict.fromkeys(str(groups[i]) for i in idx))
        cutoff = max(1, len(ordered) // 2)
        keep = set(ordered[:cutoff])
        train[idx] = np.asarray([str(groups[i]) in keep for i in idx])
        test[idx] = ~train[idx]
    return train, test


def probe(x, y, groups):
    tr, te = split(y, groups)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=700, solver="lbfgs"))
    model.fit(x[tr], y[tr])
    pred = model.predict(x[te])
    return {"accuracy": float(accuracy_score(y[te], pred)), "balanced_accuracy": float(balanced_accuracy_score(y[te], pred)), "train_epochs": int(tr.sum()), "test_epochs": int(te.sum())}


def stability(x, y, groups):
    z = StandardScaler().fit_transform(x)
    correlations, distances, within, between = [], [], [], []
    for sid in np.unique(y):
        idx = np.flatnonzero(y == sid)
        ordered = list(dict.fromkeys(str(groups[i]) for i in idx))
        first = [np.median(z[groups == key], axis=0) for key in ordered[: len(ordered) // 2]]
        second = [np.median(z[groups == key], axis=0) for key in ordered[len(ordered) // 2 :]]
        if first and second:
            a, b = np.median(first, axis=0), np.median(second, axis=0)
            correlations.append(float(np.corrcoef(a, b)[0, 1]))
            distances.append(float(np.linalg.norm(a - b)))
        center = np.median(z[idx], axis=0)
        within.extend(np.linalg.norm(z[idx] - center, axis=1).tolist())
    centers = [np.median(z[y == sid], axis=0) for sid in np.unique(y)]
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            between.append(float(np.linalg.norm(centers[i] - centers[j])))
    return {"first_second_profile_correlation_mean": float(np.mean(correlations)), "first_second_profile_distance_mean": float(np.mean(distances)), "within_epoch_radius_median": float(np.median(within)), "between_centroid_distance_median": float(np.median(between)), "between_within_ratio": float(np.median(between) / max(np.median(within), 1e-12))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.data_root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = load_base_features()
    rows = files(root)
    if not rows:
        raise RuntimeError(root)
    features, subjects, groups = [], [], []
    names = None
    for index, (sid, group, data_path, label_path) in enumerate(rows, start=1):
        data = np.load(data_path, mmap_mode="r", allow_pickle=False)
        for epoch in data:
            first = base(epoch, 100.0)
            second, extra_names = extra_features(epoch, 100.0)
            features.append(np.concatenate((first, second)))
            subjects.append(np.full(1, sid, dtype=np.int64))
            groups.append(group)
            names = [*(
                [f"log_rms_ch{c}" for c in range(8)]
                + [f"log_rms_centered_ch{c}" for c in range(8)]
                + [f"roughness_ch{c}" for c in range(8)]
                + [f"diff_ratio_ch{c}" for c in range(8)]
                + [f"relative_{b}_power_ch{c}" for b in ("delta", "theta", "alpha", "beta", "gamma") for c in range(8)]
                + [f"spectral_entropy_ch{c}" for c in range(8)]
                + ["signed_connectivity_mean", "signed_connectivity_std", "signed_connectivity_abs_mean"]
            ), *extra_names]
        if index % 10 == 0 or index == len(rows):
            print(json.dumps({"event": "sequence_done", "index": index, "total": len(rows)}), flush=True)
    x = np.asarray(features, dtype=np.float32)
    y = np.concatenate(subjects)
    g = np.asarray(groups, dtype=object)
    np.save(out / "expanded-features.npy", x)
    np.save(out / "subjects.npy", y)
    np.save(out / "groups.npy", g)
    all_result = probe(x, y, g)
    result = {"schema_version": 1, "dataset": "ISRUC-medium", "data_root": str(root), "sequence_count": len(rows), "epoch_count": int(len(y)), "subject_count": int(len(np.unique(y))), "feature_dim": int(x.shape[1]), "feature_names": names, "base_feature_dim": 83, "expanded_feature_dim": int(x.shape[1]), "identity_probe": {"all_expanded": all_result}, "stability": stability(x, y, g), "feature_groups": {"base_83": [0, 83], "extra_time_domain_hjorth": [83, 171], "extra_absolute_band_power": [171, 219], "extra_spectral_shape": [219, 259], "extra_psd_slope": [259, 267], "extra_pairwise_signed_correlation": [267, int(x.shape[1])]}, "scientific_conclusion_allowed": False}
    # Group ablations are expensive only in the small 20-subject medium set.
    for label, (start, stop) in result["feature_groups"].items():
        keep = np.ones(x.shape[1], dtype=bool)
        keep[start:stop] = False
        result["identity_probe"][f"without_{label}"] = {"removed_dimensions": stop - start, **probe(x[:, keep], y, g)}
    (out / "expanded-fingerprint-summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    standardized = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2, random_state=0).fit_transform(standardized)
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(pca[:, 0], pca[:, 1], c=y, cmap="turbo", s=10, alpha=0.55, rasterized=True)
    ax.set(title="ISRUC-medium expanded waveform fingerprint (epoch-level)", xlabel="PCA-1", ylabel="PCA-2")
    fig.colorbar(scatter, ax=ax, label="subject id")
    fig.tight_layout(); fig.savefig(out / "expanded-fingerprint-pca.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    labels = list(result["identity_probe"])
    vals = [result["identity_probe"][k]["accuracy"] for k in labels]
    ax.bar(np.arange(len(labels)), vals, color=["tab:blue"] + ["tab:orange"] * (len(labels) - 1))
    ax.axhline(1.0 / len(np.unique(y)), color="black", linestyle="--", label="20-class chance")
    ax.set_xticks(np.arange(len(labels)), [k.replace("without_", "−") for k in labels], rotation=45, ha="right")
    ax.set_ylabel("sequence-disjoint identity accuracy"); ax.set_title("Expanded feature bank ablation"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "expanded-fingerprint-ablation.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    (out / "REPORT.md").write_text("\n".join(["# Expanded ISRUC fingerprint analysis", "", f"The CPU pilot processed {len(rows)} sequences, {len(y)} epochs and {len(np.unique(y))} subjects.", "", f"The 83-D baseline was expanded to {x.shape[1]} dimensions with robust time-domain/Hjorth, absolute spectral, spectral-shape, PSD-slope and pairwise signed-correlation features.", "", f"All expanded features held-out Logistic identity accuracy: {all_result['accuracy']:.4f}; first/second subject-profile correlation: {result['stability']['first_second_profile_correlation_mean']:.4f}.", "", "Feature-group ablations are in `expanded-fingerprint-summary.json`. This is a descriptive identity-stability audit, not an LoP result or a biometric authentication claim."]) + "\n", encoding="utf-8")
    (out / "EXPLANATION.md").write_text("\n".join(["# 图表说明", "", "`expanded-fingerprint-pca.png` 每个点是一个 30 秒 epoch，颜色是 subject。它使用扩展后的波形统计向量，不是 CNN embedding；同色点的扩散表示被试内 sequence/state 漂移。", "", "`expanded-fingerprint-ablation.png` 比较完整扩展向量和去掉某个特征组后的 sequence-disjoint Logistic identity accuracy。柱子下降越多，表示该组对跨 sequence 个体区分贡献越大；它不表示生理因果重要性。", "", "扩展特征包括均值/中位数/标准差/偏度/峰度/crest factor/过零率、Hjorth activity-mobility-complexity、线长、六频带绝对 log 功率、谱质心/带宽/95%频谱边缘/平坦度/峰频、PSD 斜率和 28 个逐对带符号相关系数。具体顺序见 `expanded-fingerprint-summary.json`。", "", "扩大维度并不自动提高个体不变性：冗余特征、睡眠阶段或伪迹特征可能提高训练分数却降低跨 sequence 稳定性。因此应同时看 identity accuracy、first/second profile correlation、between/within distance ratio，并在不同 session 上复核。"]) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(out), "sequences": len(rows), "epochs": len(y), "subjects": len(np.unique(y)), "feature_dim": int(x.shape[1]), "identity_accuracy": all_result["accuracy"], "profile_corr": result["stability"]["first_second_profile_correlation_mean"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
