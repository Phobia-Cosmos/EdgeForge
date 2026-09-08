#!/usr/bin/env python3
"""Create visual diagnostics for cross-subject EEG drift.

The script reads the EdgeForge ISRUC medium subset without modifying it.  It
writes PNG figures and a compact JSON/Markdown summary covering raw waveform
scale, Welch spectra, subject-level feature PCA, and train/evaluation label
shift.  It is descriptive; the figures are not LoP evidence by themselves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


FS_HZ = 100.0
BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "sigma": (13.0, 16.0), "beta": (16.0, 30.0)}


def _integral(values: np.ndarray, coordinates: np.ndarray) -> float:
    """Integrate with NumPy 2.x ``trapezoid`` or the 1.x-compatible fallback."""
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    return float(trapezoid(values, coordinates))


def _files(root: Path, group: str, subject: int) -> list[Path]:
    return sorted((root / group / str(subject) / "data").glob("*.npy"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)


def _load_subject(root: Path, group: str, subject: int) -> tuple[np.ndarray, np.ndarray]:
    values = [np.load(path, allow_pickle=False).astype(np.float32, copy=False) for path in _files(root, group, subject)]
    labels = [np.load(root / group / str(subject) / "label" / path.name, allow_pickle=False).reshape(-1) for path in _files(root, group, subject)]
    if not values:
        raise FileNotFoundError(f"no files for {group}/{subject}")
    return np.concatenate(values, axis=0), np.concatenate(labels, axis=0).astype(np.int64, copy=False)


def _profile(root: Path, group: str, subject: int) -> dict[str, Any]:
    values, labels = _load_subject(root, group, subject)
    return {"group": group, "subject": int(subject), "values": values, "labels": labels}


def _bandpower(frequencies: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    mask = (frequencies >= low) & (frequencies < high)
    return _integral(power[mask], frequencies[mask]) if np.any(mask) else 0.0


def _mean_epoch_psd(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a channel-averaged PSD, averaged over epochs.

    Averaging channel PSDs avoids phase cancellation that can occur when the
    channels are averaged in the time domain before estimating a spectrum.
    """
    epoch_powers: list[np.ndarray] = []
    frequencies: np.ndarray | None = None
    for epoch in values:
        channel_powers = []
        for channel in epoch:
            frequencies, power = welch(channel, fs=FS_HZ, nperseg=1024)
            channel_powers.append(power)
        epoch_powers.append(np.mean(np.asarray(channel_powers), axis=0))
    if frequencies is None:
        raise ValueError("cannot estimate a spectrum from an empty subject")
    return frequencies, np.mean(np.asarray(epoch_powers), axis=0)


def _epoch_features(values: np.ndarray) -> np.ndarray:
    features: list[np.ndarray] = []
    for epoch in values:
        rms = np.sqrt(np.mean(np.square(epoch), axis=1))
        channel_powers = []
        for channel in epoch:
            frequencies, power = welch(channel, fs=FS_HZ, nperseg=512)
            channel_powers.append(power)
        power = np.mean(np.asarray(channel_powers), axis=0)
        bands = np.asarray([_bandpower(frequencies, power, low, high) for low, high in BANDS.values()], dtype=np.float64)
        features.append(np.concatenate([np.log10(rms + 1e-30), np.log10(bands + 1e-30)]))
    return np.asarray(features, dtype=np.float64)


def _label_distribution(labels: np.ndarray, classes: int = 5) -> np.ndarray:
    return np.bincount(labels, minlength=classes).astype(np.float64)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _waveform_plot(profiles: list[dict[str, Any]], output: Path, subjects: list[int]) -> None:
    selected = [item for item in profiles if item["group"] == "target" and item["subject"] in subjects]
    fig, axes = plt.subplots(len(selected), 2, figsize=(14, 2.7 * len(selected)), squeeze=False)
    time = np.arange(1000) / FS_HZ
    raw_limit = max(float(np.max(np.abs(item["values"][0, 0, :1000]))) for item in selected)
    for row, item in enumerate(selected):
        values = item["values"]
        raw = values[0, 0, :1000]
        normalized = (raw - raw.mean()) / max(raw.std(), 1e-30)
        axes[row, 0].plot(time, raw, linewidth=0.7, color="#244c7a")
        axes[row, 0].set_ylim(-raw_limit, raw_limit)
        subject_rms = float(np.sqrt(np.mean(np.square(values))))
        axes[row, 0].set_title(f"target subject {item['subject']} · raw channel 0 · RMS {subject_rms:.2e}")
        axes[row, 0].set_xlabel("time (s)")
        axes[row, 0].set_ylabel("amplitude (common scale)")
        axes[row, 1].plot(time, normalized, linewidth=0.7, color="#b54b4b")
        axes[row, 1].set_title(f"target subject {item['subject']} · per-epoch standardized")
        axes[row, 1].set_xlabel("time (s)")
        axes[row, 1].set_ylabel("z-score")
        axes[row, 1].set_ylim(-3.5, 3.5)
    fig.suptitle("EEG waveform scale drift across target subjects (raw panels share y-scale)", y=1.01, fontsize=14)
    _save(fig, output / "eeg-drift-waveforms.png")


def _normalized_overlay_plot(profiles: list[dict[str, Any]], output: Path, subjects: list[int]) -> None:
    selected = [item for item in profiles if item["group"] == "target" and item["subject"] in subjects]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    time = np.arange(1000) / FS_HZ
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(selected)))
    for color, item in zip(colors, selected):
        epochs = item["values"][:, 0, :1000].astype(np.float64, copy=False)
        # Pick the epoch whose RMS is closest to the subject median. Averaging
        # epochs would cancel phase-varying EEG activity and create an
        # artificial flat line; this deterministic representative retains a
        # real waveform while avoiding an arbitrary first-epoch choice.
        epoch_rms = np.sqrt(np.mean(np.square(epochs), axis=1))
        representative = epochs[int(np.argmin(np.abs(epoch_rms - np.median(epoch_rms))))]
        normalized = (representative - representative.mean()) / max(representative.std(), 1e-30)
        window = 5
        smoothed = np.convolve(normalized, np.ones(window) / window, mode="same")
        ax.plot(time, smoothed, linewidth=1.2, alpha=0.85, color=color, label=f"subject {item['subject']}")
    ax.set_title("Typical normalized waveform morphology across target subjects")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("channel-0 z-score (median-RMS epoch, smoothed)")
    ax.set_ylim(-3.5, 3.5)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    _save(fig, output / "eeg-drift-normalized-overlay.png")


def _spectra_plot(profiles: list[dict[str, Any]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len([p for p in profiles if p["group"] == "target"])))
    color_index = 0
    for item in profiles:
        if item["group"] != "target":
            continue
        frequencies, power = _mean_epoch_psd(item["values"])
        mask = (frequencies >= 0.5) & (frequencies <= 40.0)
        ax.plot(frequencies[mask], 10.0 * np.log10(power[mask] + 1e-30), linewidth=1.2, color=colors[color_index], label=f"subject {item['subject']}")
        color_index += 1
    ax.set_title("Mean per-epoch channel-averaged Welch spectrum across target subjects")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("power (dB, relative scale)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    _save(fig, output / "eeg-drift-spectra.png")


def _rms_trajectory_plot(profiles: list[dict[str, Any]], output: Path, subjects: list[int]) -> None:
    selected = [item for item in profiles if item["group"] == "target" and item["subject"] in subjects]
    trajectories = [np.sqrt(np.mean(np.square(item["values"]), axis=(1, 2))) for item in selected]
    # ISRUC subjects do not all contain the same number of 20-epoch files.
    # Pad only the visualization matrix with NaN so shorter streams keep their
    # true length and are not silently repeated or truncated.
    max_length = max((len(values) for values in trajectories), default=0)
    matrix = np.full((len(trajectories), max_length), np.nan, dtype=np.float64)
    for row, values in enumerate(trajectories):
        matrix[row, : len(values)] = values
    log_matrix = np.log10(matrix + 1e-30)
    fig, ax = plt.subplots(figsize=(13, 4.8))
    image = ax.imshow(log_matrix, aspect="auto", interpolation="nearest", cmap="viridis", origin="lower")
    if matrix.shape[1]:
        ax.axvline(matrix.shape[1] / 2 - 0.5, color="white", linewidth=1.2, linestyle="--", alpha=0.9)
    ax.set_title("EEG amplitude drift over each target subject stream")
    ax.set_xlabel("epoch index (dashed line: adaptation/evaluation split)")
    ax.set_ylabel("target subject")
    ax.set_yticks(range(len(selected)), [str(item["subject"]) for item in selected])
    fig.colorbar(image, ax=ax, label="log10(epoch RMS)")
    _save(fig, output / "eeg-drift-rms-trajectory.png")


def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = (left + 1e-30) / (left.sum() + 1e-30 * len(left))
    right = (right + 1e-30) / (right.sum() + 1e-30 * len(right))
    midpoint = 0.5 * (left + right)
    return float(0.5 * np.sum(left * np.log(left / midpoint)) + 0.5 * np.sum(right * np.log(right / midpoint)))


def _quantify_drift(profiles: list[dict[str, Any]], subjects: list[int]) -> dict[str, Any]:
    selected = [item for item in profiles if item["group"] == "target" and item["subject"] in subjects]
    rms_by_subject: dict[str, float] = {}
    band_fraction_by_subject: dict[str, dict[str, float]] = {}
    normalized_psds = []
    for item in selected:
        subject = str(item["subject"])
        rms_by_subject[subject] = float(np.sqrt(np.mean(np.square(item["values"]))))
        frequencies, power = _mean_epoch_psd(item["values"])
        total = max(_bandpower(frequencies, power, 0.5, 40.0), 1e-30)
        band_fraction_by_subject[subject] = {
            name: _bandpower(frequencies, power, low, high) / total
            for name, (low, high) in BANDS.items()
        }
        mask = (frequencies >= 0.5) & (frequencies <= 40.0)
        normalized_psds.append(power[mask] / max(_integral(power[mask], frequencies[mask]), 1e-30))
    median_psd = np.median(np.asarray(normalized_psds), axis=0)
    median_psd = median_psd / max(float(median_psd.sum()), 1e-30)
    spectral_js_to_median = {
        str(item["subject"]): _js_divergence(normalized_psds[index], median_psd)
        for index, item in enumerate(selected)
    }
    rms_values = list(rms_by_subject.values())
    feature_means = []
    feature_subjects = []
    for item in selected:
        feature_means.append(np.mean(_epoch_features(item["values"]), axis=0))
        feature_subjects.append(str(item["subject"]))
    feature_matrix = StandardScaler().fit_transform(np.asarray(feature_means, dtype=np.float64))
    pairwise = np.sqrt(np.maximum(0.0, ((feature_matrix[:, None, :] - feature_matrix[None, :, :]) ** 2).sum(axis=-1)))
    upper = [(float(pairwise[i, j]), feature_subjects[i], feature_subjects[j]) for i in range(len(feature_subjects)) for j in range(i + 1, len(feature_subjects))]
    farthest = max(upper) if upper else (0.0, None, None)
    return {
        "target_rms_by_subject": rms_by_subject,
        "target_rms_max_min_ratio": float(max(rms_values) / max(min(rms_values), 1e-30)),
        "target_rms_log10_span": float(np.log10(max(rms_values)) - np.log10(max(min(rms_values), 1e-30))),
        "target_band_fraction_by_subject": band_fraction_by_subject,
        "target_spectral_js_to_median": spectral_js_to_median,
        "target_feature_subject_order": feature_subjects,
        "target_feature_pairwise_euclidean": pairwise.tolist(),
        "target_feature_farthest_pair": {"subject_a": farthest[1], "subject_b": farthest[2], "distance": farthest[0]},
        "spectral_distance_definition": "Jensen-Shannon divergence of normalized 0.5-40 Hz mean channel PSD to the target-subject median PSD",
        "feature_distance_definition": "Euclidean distance after standardizing subject means of epoch-level log-RMS and band-power features across target subjects",
    }


def _subject_distance_plot(profiles: list[dict[str, Any]], output: Path, subjects: list[int]) -> dict[str, Any]:
    selected = [item for item in profiles if item["group"] == "target" and item["subject"] in subjects]
    subject_labels = [str(item["subject"]) for item in selected]
    feature_means = np.asarray([np.mean(_epoch_features(item["values"]), axis=0) for item in selected], dtype=np.float64)
    feature_matrix = StandardScaler().fit_transform(feature_means)
    distances = np.sqrt(np.maximum(0.0, ((feature_matrix[:, None, :] - feature_matrix[None, :, :]) ** 2).sum(axis=-1)))
    fig, ax = plt.subplots(figsize=(8, 6.5))
    image = ax.imshow(distances, cmap="magma", interpolation="nearest")
    ax.set_title("Pairwise target-subject distance in EEG feature space")
    ax.set_xlabel("target subject")
    ax.set_ylabel("target subject")
    ax.set_xticks(range(len(subject_labels)), subject_labels)
    ax.set_yticks(range(len(subject_labels)), subject_labels)
    fig.colorbar(image, ax=ax, label="standardized feature distance")
    _save(fig, output / "eeg-drift-subject-distance.png")
    return {"subjects": subject_labels, "matrix": distances.tolist()}


def _pca_plot(profiles: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    rows: list[np.ndarray] = []
    metadata: list[tuple[str, int]] = []
    for item in profiles:
        features = _epoch_features(item["values"])
        # Keep a deterministic subset so the figure is easy to regenerate.
        rows.append(features[:: max(1, len(features) // 40)])
        metadata.extend([(item["group"], item["subject"])] * len(rows[-1]))
    matrix = np.concatenate(rows, axis=0)
    transformed = PCA(n_components=2, random_state=0).fit_transform(StandardScaler().fit_transform(matrix))
    fig, ax = plt.subplots(figsize=(11, 8))
    roles = {"source": "#3b82f6", "target": "#ef4444", "retention": "#10b981"}
    for role, color in roles.items():
        for subject in sorted({subject for group, subject in metadata if group == role}):
            mask = np.asarray([(group == role and current_subject == subject) for group, current_subject in metadata])
            ax.scatter(transformed[mask, 0], transformed[mask, 1], s=14, alpha=0.55, color=color, label=f"{role} {subject}")
    ax.set_title("PCA of epoch-level log-RMS and band-power features")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    ax.legend(ncol=3, fontsize=7, markerscale=1.2)
    _save(fig, output / "eeg-drift-pca.png")
    explained = PCA(n_components=2, random_state=0).fit(StandardScaler().fit_transform(matrix)).explained_variance_ratio_
    return {"rows": int(matrix.shape[0]), "features": int(matrix.shape[1]), "explained_variance_ratio": explained.tolist()}


def _label_plot(profiles: list[dict[str, Any]], output: Path, classes: int = 5) -> None:
    target = [item for item in profiles if item["group"] == "target"]
    matrix: list[np.ndarray] = []
    labels: list[str] = []
    for item in target:
        y = item["labels"]
        matrix.append(_label_distribution(y[: len(y) // 2], classes) / max(1, len(y) // 2))
        matrix.append(_label_distribution(y[len(y) // 2 :], classes) / max(1, len(y) // 2))
        labels.extend([f"{item['subject']} adapt", f"{item['subject']} eval"])
    values = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(13, 6))
    image = ax.imshow(values, aspect="auto", cmap="magma", vmin=0.0, vmax=0.6)
    ax.set_title("Target label prior drift: adaptation half vs held-out half")
    ax.set_xlabel("sleep-stage class")
    ax.set_ylabel("target subject / split")
    ax.set_xticks(range(classes), [str(i) for i in range(classes)])
    ax.set_yticks(range(len(labels)), labels)
    fig.colorbar(image, ax=ax, label="class fraction")
    _save(fig, output / "eeg-drift-labels.png")


def visualize(
    data_root: str | Path,
    output_dir: str | Path,
    target_subjects: list[int],
    *,
    source_subjects: list[int] | None = None,
    retention_subjects: list[int] | None = None,
) -> dict[str, Any]:
    root = Path(data_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    profiles = []
    groups = {
        "source": list(source_subjects if source_subjects is not None else [1, 3, 4, 6, 7, 9, 10, 21]),
        "target": list(target_subjects),
        "retention": list(retention_subjects if retention_subjects is not None else [5, 18, 19, 20]),
    }
    for group, subjects in groups.items():
        for subject in subjects:
            profiles.append(_profile(root, group, subject))
    _waveform_plot(profiles, output, target_subjects)
    _normalized_overlay_plot(profiles, output, target_subjects)
    _spectra_plot(profiles, output)
    _rms_trajectory_plot(profiles, output, target_subjects)
    pca_info = _pca_plot(profiles, output)
    _label_plot(profiles, output)
    quantitative = _quantify_drift(profiles, target_subjects)
    distance = _subject_distance_plot(profiles, output, target_subjects)
    quantitative["target_feature_pairwise_euclidean"] = distance["matrix"]
    summary = {"schema_version": 1, "analysis": "eeg-drift-visualization-v1", "data_root": str(root), "output_dir": str(output), "sampling_rate_hz": FS_HZ, "groups": groups, "pca": pca_info, "quantitative_drift": quantitative, "figures": ["eeg-drift-waveforms.png", "eeg-drift-normalized-overlay.png", "eeg-drift-spectra.png", "eeg-drift-rms-trajectory.png", "eeg-drift-subject-distance.png", "eeg-drift-pca.png", "eeg-drift-labels.png"], "scientific_conclusion_allowed": False}
    (output / "visualization-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text("# EEG drift visualizations\n\nThese figures visualize signal-scale, spectral, feature-space, within-stream and label-prior drift in the ISRUC medium development subset. They are descriptive diagnostics, not LoP evidence.\n\n- `eeg-drift-waveforms.png`: raw channel-0 waveforms on a common y-scale (RMS is shown in each title) alongside per-epoch standardized shape.\n- `eeg-drift-normalized-overlay.png`: typical normalized waveform morphology (the epoch closest to each subject's median RMS) overlaid across subjects after removing gain.\n- `eeg-drift-spectra.png`: mean per-epoch, channel-averaged Welch power spectra for target subjects.\n- `eeg-drift-rms-trajectory.png`: epoch-level RMS heatmap; the dashed line separates adaptation and held-out halves.\n- `eeg-drift-subject-distance.png`: pairwise standardized feature distance between target subjects.\n- `eeg-drift-pca.png`: PCA of log-RMS and band-power features.\n- `eeg-drift-labels.png`: adaptation/evaluation label fractions.\n- `visualization-summary.json`: reproducible paths and quantitative RMS/spectral/feature-drift summaries.\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-subject", action="append", type=int, default=[], help="legacy repeated target subject option")
    parser.add_argument("--source-subjects", nargs="+", type=int, default=None)
    parser.add_argument("--target-subjects", nargs="+", type=int, default=None)
    parser.add_argument("--retention-subjects", nargs="+", type=int, default=None)
    args = parser.parse_args()
    targets = args.target_subjects or args.target_subject or [2, 11, 12, 13, 14, 15, 16, 17]
    summary = visualize(
        args.data_root,
        args.output_dir,
        targets,
        source_subjects=args.source_subjects,
        retention_subjects=args.retention_subjects,
    )
    print(json.dumps({"status": "ok", "figures": summary["figures"], "output_dir": str(args.output_dir.resolve()), "scientific_conclusion_allowed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
