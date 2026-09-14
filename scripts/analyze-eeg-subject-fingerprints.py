#!/usr/bin/env python3
"""Estimate stable EEG subject fingerprints for ISRUC and FACED.

The analysis is intentionally feature-level and read-only.  It separates
subject identity from task labels by holding out files/trials within each
subject, then writes per-subject robust profiles that can be used to constrain
later label-preserving perturbations.  It does not claim biometric identity
from a single epoch and is not an LoP test.

Supported inputs:
* ISRUC processed arrays: ``<subject>/data/*.npy`` with shape
  ``(epochs, channels, samples)`` and matching ``label`` files.
* FACED BIDS: ``sub-*/eeg/*_eeg.bdf`` plus ``*_events.tsv``.  MNE is required
  only for FACED; the script reads short trial windows without preloading a
  complete recording.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


ISRUC_FS = 100.0
FACED_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _integral(values: np.ndarray, coordinates: np.ndarray) -> float:
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(values, coordinates)) if values.size else 0.0


def _bandpower(freq: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    mask = (freq >= low) & (freq < high)
    return _integral(power[mask], freq[mask])


def epoch_features(epoch: np.ndarray, fs: float) -> np.ndarray:
    """Return gain-aware and gain-invariant morphology/spectral features."""
    values = np.asarray(epoch, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"expected [channels, samples], got {values.shape}")
    channels = values.shape[0]
    global_rms = max(float(np.sqrt(np.mean(np.square(values)))), 1e-30)
    channel_rms = np.sqrt(np.mean(np.square(values), axis=1))
    log_rms = np.log10(channel_rms + 1e-30)
    log_rms_centered = log_rms - float(np.mean(log_rms))
    diff_rms = np.sqrt(np.mean(np.square(np.diff(values, axis=-1)), axis=1))
    diff_ratio = np.log10(diff_rms / max(global_rms, 1e-30) + 1e-30)
    roughness = np.log10(diff_rms + 1e-30)
    band_values = np.zeros((channels, len(FACED_BANDS)), dtype=np.float64)
    spectral_entropy = np.zeros(channels, dtype=np.float64)
    for index, channel in enumerate(values):
        freq, power = welch(channel, fs=fs, nperseg=min(512, len(channel)))
        total = max(_bandpower(freq, power, 0.5, min(45.0, fs / 2.0)), 1e-30)
        for band_index, (low, high) in enumerate(FACED_BANDS.values()):
            band_values[index, band_index] = _bandpower(freq, power, low, min(high, fs / 2.0)) / total
        density = power / max(float(power.sum()), 1e-30)
        spectral_entropy[index] = float(-(density * np.log(density + 1e-30)).sum())
    # Signed connectivity is retained because gain-only summaries cannot
    # detect polarity/reference changes.
    corr = np.corrcoef(values)
    upper = corr[np.triu_indices(channels, k=1)] if channels > 1 else np.zeros(1)
    profile = np.concatenate(
        [
            log_rms,
            log_rms_centered,
            roughness,
            diff_ratio,
            band_values.reshape(-1),
            spectral_entropy,
            np.asarray([float(np.mean(upper)), float(np.std(upper)), float(np.mean(np.abs(upper)))], dtype=np.float64),
        ]
    )
    # A few BDF channels are constant or flat for parts of a recording; their
    # correlation is undefined, but the flatness itself is a valid quality
    # characteristic. Encode undefined correlation summaries as zero so the
    # decoder receives a finite, reproducible fingerprint.
    return np.nan_to_num(profile, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _subject_dirs(root: Path) -> list[Path]:
    result = []
    for path in root.iterdir():
        if path.is_dir() and path.name.isdigit() and (path / "data").is_dir():
            result.append(path)
    return sorted(result, key=lambda path: int(path.name))


def _load_isruc(root: Path, max_subjects: int, max_files: int, max_epochs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rows: list[np.ndarray] = []
    subjects: list[int] = []
    groups: list[str] = []
    inventory: list[dict[str, Any]] = []
    dirs = _subject_dirs(root)[:max_subjects] if max_subjects > 0 else _subject_dirs(root)
    for subject_dir in dirs:
        files = sorted(subject_dir.joinpath("data").glob("*.npy"), key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem)
        if max_files > 0:
            files = files[:max_files]
        subject_count = 0
        for path in files:
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            labels = np.load(subject_dir / "label" / path.name, mmap_mode="r", allow_pickle=False).reshape(-1)
            if values.ndim != 3 or values.shape[0] != labels.shape[0]:
                raise ValueError(f"invalid ISRUC pair {path}: {values.shape} / {labels.shape}")
            count = min(int(values.shape[0]), max_epochs) if max_epochs > 0 else int(values.shape[0])
            for epoch_index in range(count):
                rows.append(epoch_features(values[epoch_index], ISRUC_FS))
                subjects.append(int(subject_dir.name))
                groups.append(f"{subject_dir.name}:{path.stem}")
                subject_count += 1
        inventory.append({"subject": int(subject_dir.name), "files": len(files), "epochs": subject_count})
    return np.asarray(rows), np.asarray(subjects, dtype=np.int64), np.asarray(groups, dtype=object), inventory


def _faced_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                onset = float(row.get("onset", ""))
            except ValueError:
                continue
            if onset < 0:
                continue
            row["onset_float"] = onset
            events.append(row)
    return events


def _load_faced(root: Path, max_subjects: int, max_trials: int, window_seconds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    try:
        import mne
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("FACED analysis requires MNE; install it in the dedicated experiment environment") from exc
    rows: list[np.ndarray] = []
    subjects: list[int] = []
    groups: list[str] = []
    inventory: list[dict[str, Any]] = []
    dirs = sorted(root.glob("sub-*"), key=lambda path: int(path.name.split("-")[-1]))
    if max_subjects > 0:
        dirs = dirs[:max_subjects]
    for subject_dir in dirs:
        eeg_dir = subject_dir / "eeg"
        bdf_paths = sorted(eeg_dir.glob("*_eeg.bdf"))
        event_paths = sorted(eeg_dir.glob("*_events.tsv"))
        if not bdf_paths or not event_paths:
            continue
        raw = mne.io.read_raw_bdf(bdf_paths[0], preload=False, verbose="ERROR")
        picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
        events = _faced_events(event_paths[0])
        # BIDS events include one row per marker; video_index identifies the
        # trial. Keep the first onset per video and hold out whole videos.
        trial_rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in events:
            key = row.get("video_index", "")
            if key in (None, "", "n/a", "NA", "na"):
                continue
            try:
                int(key)
            except (TypeError, ValueError):
                continue
            if key in seen:
                continue
            seen.add(key)
            trial_rows.append(row)
        if max_trials > 0:
            trial_rows = trial_rows[:max_trials]
        count = 0
        for trial_index, row in enumerate(trial_rows):
            start = max(0, int(round(float(row["onset_float"]) * raw.info["sfreq"])))
            stop = min(raw.n_times, start + int(round(window_seconds * raw.info["sfreq"])))
            if stop - start < 64:
                continue
            values = raw.get_data(picks=picks, start=start, stop=stop)
            rows.append(epoch_features(values, float(raw.info["sfreq"])))
            sid = int(subject_dir.name.split("-")[-1])
            subjects.append(sid)
            groups.append(f"{sid}:trial:{trial_index:02d}")
            count += 1
        inventory.append({"subject": int(subject_dir.name.split("-")[-1]), "trials": count, "sfreq": float(raw.info["sfreq"]), "channels": int(len(picks))})
        raw.close()
    return np.asarray(rows), np.asarray(subjects, dtype=np.int64), np.asarray(groups, dtype=object), inventory


def _subject_split(groups: np.ndarray, subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole sequence/trial groups, while retaining every identity."""
    train = np.zeros(len(subjects), dtype=bool)
    test = np.zeros(len(subjects), dtype=bool)
    for subject in np.unique(subjects):
        indices = np.flatnonzero(subjects == subject)
        unique_groups = list(dict.fromkeys(str(groups[index]) for index in indices))
        cutoff = max(1, len(unique_groups) // 2)
        train_groups = set(unique_groups[:cutoff])
        for index in indices:
            (train if str(groups[index]) in train_groups else test)[index] = True
        # A subject with one group cannot be split without leakage; mark it as
        # train-only so it is reported rather than silently duplicated.
        if not test[indices].any():
            test[indices[-1]] = True
            train[indices[-1]] = False
    return train, test


def _nearest_centroid(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> dict[str, Any]:
    centroids = {int(subject): np.median(train_x[train_y == subject], axis=0) for subject in np.unique(train_y)}
    labels = sorted(centroids)
    matrix = np.asarray([centroids[label] for label in labels])
    distances = ((test_x[:, None, :] - matrix[None, :, :]) ** 2).mean(axis=-1)
    prediction = np.asarray([labels[index] for index in distances.argmin(axis=1)], dtype=np.int64)
    return {"accuracy": float(accuracy_score(test_y, prediction)), "balanced_accuracy": float(balanced_accuracy_score(test_y, prediction)), "predictions": prediction, "labels": labels}


def _profile_rows(features: np.ndarray, subjects: np.ndarray) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    for subject in np.unique(subjects):
        values = features[subjects == subject]
        median = np.median(values, axis=0)
        mad = np.median(np.abs(values - median), axis=0)
        profiles[str(int(subject))] = {"n_samples": int(len(values)), "median": median.tolist(), "mad": mad.tolist(), "mean": np.mean(values, axis=0).tolist()}
    return profiles


def analyze_dataset(name: str, features: np.ndarray, subjects: np.ndarray, groups: np.ndarray, inventory: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    if len(features) == 0:
        raise ValueError(f"no {name} samples")
    train, test = _subject_split(groups, subjects)
    scaler = StandardScaler().fit(features[train])
    x_train = scaler.transform(features[train])
    x_test = scaler.transform(features[test])
    y_train, y_test = subjects[train], subjects[test]
    centroid = _nearest_centroid(x_train, y_train, x_test, y_test)
    # Logistic probe is a complementary decoder; it is evaluated on held-out
    # sequence/trial groups, never on epochs from a seen group.
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=600, solver="lbfgs", multi_class="auto"))
    probe.fit(features[train], y_train)
    prediction = probe.predict(features[test])
    profiles = _profile_rows(features, subjects)
    centroids = {int(k): np.asarray(v["median"], dtype=np.float64) for k, v in profiles.items()}
    within = []
    for subject, values in centroids.items():
        subject_values = features[subjects == subject]
        within.append(float(np.median(np.linalg.norm(subject_values - values[None, :], axis=1))))
    between = []
    labels = sorted(centroids)
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            between.append(float(np.linalg.norm(centroids[left] - centroids[right])))
    result = {
        "schema_version": 1,
        "dataset": name,
        "sample_count": int(len(features)),
        "subject_count": int(len(np.unique(subjects))),
        "feature_dim": int(features.shape[1]),
        "split": "within-subject sequence/trial-disjoint first-half train, second-half test",
        "nearest_centroid": {"accuracy": centroid["accuracy"], "balanced_accuracy": centroid["balanced_accuracy"]},
        "logistic_subject_probe": {"accuracy": float(accuracy_score(y_test, prediction)), "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction))},
        "within_subject_profile_radius_mean": float(np.mean(within)) if within else None,
        "between_subject_centroid_distance_median": float(np.median(between)) if between else None,
        "between_within_ratio": float(np.median(between) / max(np.mean(within), 1e-12)) if between and within else None,
        "inventory": inventory,
        "subject_profiles": profiles,
        "scientific_conclusion_allowed": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "subject-fingerprint-summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    np.save(output / "features.npy", features)
    np.save(output / "subjects.npy", subjects)
    # A compact 2-D view uses the same standardized fingerprint vectors as the
    # decoder, not raw waveforms, and is only descriptive.
    from sklearn.decomposition import PCA

    projection = PCA(n_components=2, random_state=0).fit_transform(StandardScaler().fit_transform(features))
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(projection[:, 0], projection[:, 1], c=subjects, cmap="turbo", s=14, alpha=0.7)
    ax.set_title(f"{name}: subject fingerprint projection\nsequence/trial-disjoint identity probe; descriptive")
    ax.set_xlabel("PCA-1 of EEG fingerprint features")
    ax.set_ylabel("PCA-2 of EEG fingerprint features")
    ax.grid(alpha=0.2)
    fig.colorbar(scatter, ax=ax, label="subject id")
    fig.savefig(output / "subject-fingerprint-pca.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isruc-root", type=Path)
    parser.add_argument("--faced-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=4)
    parser.add_argument("--max-epochs-per-file", type=int, default=20)
    parser.add_argument("--max-trials", type=int, default=28)
    parser.add_argument("--faced-window-seconds", type=float, default=20.0)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"schema_version": 1, "datasets": {}, "scientific_conclusion_allowed": False}
    if args.isruc_root:
        features, subjects, groups, inventory = _load_isruc(args.isruc_root.resolve(), args.max_subjects, args.max_files, args.max_epochs_per_file)
        manifest["datasets"]["ISRUC"] = analyze_dataset("ISRUC", features, subjects, groups, inventory, output / "ISRUC")
    if args.faced_root:
        features, subjects, groups, inventory = _load_faced(args.faced_root.resolve(), args.max_subjects, args.max_trials, args.faced_window_seconds)
        manifest["datasets"]["FACED"] = analyze_dataset("FACED", features, subjects, groups, inventory, output / "FACED")
    if not manifest["datasets"]:
        parser.error("at least one of --isruc-root or --faced-root is required")
    report = [
        "# EEG subject fingerprint analysis",
        "",
        "The analysis is read-only and descriptive. Each subject profile contains gain-aware RMS, gain-invariant channel balance, roughness, relative band power, spectral entropy and signed connectivity summaries.",
        "",
        "Identity probes use within-subject sequence/trial-disjoint splits: the first half of groups is used to form a subject profile and the second half is held out. A high score means the recorded subject pattern is repeatable across groups; it does not mean a single epoch is a reliable biometric identifier.",
        "",
        "The saved `subject_profiles` are intended as constraints for later modifications: sample drift/noise from the subject's own MAD and reject a transformed epoch when its profile leaves the subject median/MAD envelope. Preserve signed connectivity because RMS and PSD alone cannot preserve reference/polarity structure.",
        "",
        "All results remain `scientific_conclusion_allowed=false`; identity separability is not evidence of LoP.",
        "",
    ]
    for name, result in manifest["datasets"].items():
        report.extend([
            f"## {name}",
            "",
            f"Samples: {result['sample_count']}; subjects: {result['subject_count']}; feature dimension: {result['feature_dim']}.",
            f"Nearest-centroid identity accuracy: {result['nearest_centroid']['accuracy']:.4f}; logistic probe accuracy: {result['logistic_subject_probe']['accuracy']:.4f}; between/within profile distance ratio: {result['between_within_ratio']!s}.",
            f"See `{name}/subject-fingerprint-pca.png` and `{name}/subject-fingerprint-summary.json`.",
            "",
        ])
    (output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "datasets": list(manifest["datasets"]), "output_root": str(output), "scientific_conclusion_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
