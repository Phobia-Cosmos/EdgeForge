#!/usr/bin/env python3
"""Create label-preserving EEG perturbation conditions outside the checkout.

The source ISRUC subset is never modified.  Each output keeps the original
directory contract and labels, writes a manifest with the transformation
parameters, and includes utility checks (RMS ratio, correlation and SNR).

Conditions:

``rms_equalized``
    Multiply every subject by one scalar so its global RMS equals the median
    source-subject RMS.  This removes acquisition-gain differences while
    preserving waveform shape and spectral proportions.

``target_snr20_noise``
    Leave source/retention untouched.  Add deterministic 0.5--30 Hz noise to
    target epochs at 20 dB signal-to-noise ratio.  Labels and epoch boundaries
    are unchanged; the perturbation is intended as a mild sensor-domain shift.

The related `target_snr15_noise` and `target_snr10_noise` choices provide a
pre-registered dose scan. `target_snr10_noise` is a stress condition because
its expected waveform correlation is below the preferred 0.98 utility gate.

``target_gain_drift10``
    Apply a deterministic smooth +/-10% gain envelope independently to each
    target epoch. This models slow electrode/amplifier gain drift without
    changing labels, channel order or epoch boundaries.

``target_crosstalk5`` / ``target_crosstalk10``
    Mix a small fraction of each channel with its adjacent physical channels.
    The channel axis is retained and the row-stochastic mixing matrix is
    recorded in the manifest. These conditions model sensor montage leakage,
    not a channel permutation.

``target_baseline_drift5`` / ``target_baseline_drift10``
    Add a deterministic 0.1 Hz sinusoidal baseline component at 5% or 10% of
    each epoch/channel RMS. The component is zero-mean over the 30-second
    epoch and models low-frequency acquisition drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt


FS_HZ = 100.0
NOISE_SNR_DB = 20.0
NOISE_LOW_HZ = 0.5
NOISE_HIGH_HZ = 30.0
NOISE_CONDITIONS = {
    "target_snr20_noise": 20.0,
    "target_snr15_noise": 15.0,
    "target_snr10_noise": 10.0,
}
GAIN_DRIFT_FRACTION = 0.10
GAIN_DRIFT_PERIOD_SECONDS = 30.0
CROSSTALK_CONDITIONS = {
    "target_crosstalk5": 0.05,
    "target_crosstalk10": 0.10,
}
BASELINE_DRIFT_CONDITIONS = {
    "target_baseline_drift5": 0.05,
    "target_baseline_drift10": 0.10,
}
BASELINE_DRIFT_FREQUENCY_HZ = 0.1
SUPPORTED_CONDITIONS = {
    "rms_equalized",
    "target_gain_drift10",
    *NOISE_CONDITIONS,
    *CROSSTALK_CONDITIONS,
    *BASELINE_DRIFT_CONDITIONS,
}


def _files(root: Path, group: str, subject: int) -> list[Path]:
    return sorted(
        (root / group / str(subject) / "data").glob("*.npy"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )


def _subjects(root: Path, group: str) -> list[int]:
    return sorted(int(path.name) for path in (root / group).iterdir() if path.is_dir() and path.name.isdigit())


def _subject_rms(root: Path, group: str, subject: int) -> float:
    sum_squares = 0.0
    count = 0
    for path in _files(root, group, subject):
        values = np.load(path, allow_pickle=False).astype(np.float64, copy=False)
        sum_squares += float(np.square(values).sum())
        count += int(values.size)
    if count == 0:
        raise ValueError(f"empty subject {group}/{subject}")
    return float(np.sqrt(sum_squares / count))


def _seed_for(path: Path, seed: int) -> int:
    digest = hashlib.sha256(str(path).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") ^ int(seed)


def _band_limited_noise(shape: tuple[int, ...], signal_rms: float, seed: int, snr_db: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(shape).astype(np.float32)
    sos = butter(4, [NOISE_LOW_HZ, NOISE_HIGH_HZ], btype="bandpass", fs=FS_HZ, output="sos")
    noise = sosfiltfilt(sos, noise, axis=-1).astype(np.float32, copy=False)
    noise_rms = float(np.sqrt(np.mean(np.square(noise))))
    target_noise_rms = float(signal_rms / (10.0 ** (snr_db / 20.0)))
    return noise * (target_noise_rms / max(noise_rms, 1e-30))


def _gain_drift(values: np.ndarray, seed: int) -> np.ndarray:
    """Apply a smooth, zero-mean +/-10% gain envelope to each epoch."""
    samples = int(values.shape[-1])
    time = np.arange(samples, dtype=np.float32) / FS_HZ
    phase = float(np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi))
    envelope = 1.0 + GAIN_DRIFT_FRACTION * np.sin(2.0 * np.pi * time / GAIN_DRIFT_PERIOD_SECONDS + phase)
    return values * envelope.reshape((1,) * (values.ndim - 1) + (samples,))


def _channel_mixing_matrix(channels: int, fraction: float) -> np.ndarray:
    """Return an adjacent-channel, row-stochastic cross-talk matrix."""
    channels = int(channels)
    fraction = float(fraction)
    if channels < 1:
        raise ValueError("channels must be positive")
    if not 0.0 <= fraction < 1.0:
        raise ValueError("cross-talk fraction must be in [0, 1)")
    matrix = np.zeros((channels, channels), dtype=np.float32)
    for index in range(channels):
        neighbors = [candidate for candidate in (index - 1, index + 1) if 0 <= candidate < channels]
        matrix[index, index] = np.float32(1.0 - fraction)
        if neighbors:
            matrix[index, neighbors] = np.float32(fraction / len(neighbors))
        else:  # pragma: no cover - channels == 1 is not used by ISRUC
            matrix[index, index] = 1.0
    return matrix


def _channel_crosstalk(values: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Mix adjacent channels while preserving epoch/sample coordinates."""
    if values.ndim != 3:
        raise ValueError(f"expected [epochs, channels, samples], got {values.shape}")
    matrix = _channel_mixing_matrix(values.shape[1], fraction)
    transformed = np.einsum("ij,ejt->eit", matrix, values, optimize=True)
    return transformed.astype(np.float32, copy=False), matrix


def _baseline_drift(values: np.ndarray, seed: int, fraction: float) -> np.ndarray:
    """Add a zero-mean 0.1 Hz baseline component scaled per epoch/channel."""
    if values.ndim != 3:
        raise ValueError(f"expected [epochs, channels, samples], got {values.shape}")
    fraction = float(fraction)
    if fraction < 0.0:
        raise ValueError("baseline drift fraction must be non-negative")
    epochs, channels, samples = (int(item) for item in values.shape)
    time = np.arange(samples, dtype=np.float32) / FS_HZ
    phases = np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, size=(epochs, channels)).astype(np.float32)
    channel_rms = np.sqrt(np.mean(np.square(values.astype(np.float64, copy=False)), axis=-1, keepdims=True)).astype(np.float32)
    waveform = np.sin(
        2.0 * np.pi * BASELINE_DRIFT_FREQUENCY_HZ * time.reshape(1, 1, -1)
        + phases.reshape(epochs, channels, 1)
    ).astype(np.float32, copy=False)
    return values + np.float32(fraction) * channel_rms * waveform


def _copy_or_transform(
    path: Path,
    output_path: Path,
    group: str,
    subject: int,
    condition: str,
    scale: float,
    seed: int,
    noise_snr_db: float,
) -> dict[str, Any]:
    values = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
    original = values.astype(np.float64, copy=False)
    if condition == "rms_equalized":
        transformed = values * np.float32(scale)
    elif condition in NOISE_CONDITIONS and group == "target":
        signal_rms = float(np.sqrt(np.mean(np.square(original))))
        noise = _band_limited_noise(values.shape, signal_rms, _seed_for(path, seed), noise_snr_db)
        transformed = values + noise
    elif condition == "target_gain_drift10" and group == "target":
        transformed = _gain_drift(values, _seed_for(path, seed))
    elif condition in CROSSTALK_CONDITIONS and group == "target":
        transformed, _matrix = _channel_crosstalk(values, CROSSTALK_CONDITIONS[condition])
    elif condition in BASELINE_DRIFT_CONDITIONS and group == "target":
        transformed = _baseline_drift(values, _seed_for(path, seed), BASELINE_DRIFT_CONDITIONS[condition])
    else:
        transformed = values.copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, transformed.astype(np.float32, copy=False), allow_pickle=False)
    difference = transformed.astype(np.float64, copy=False) - original
    original_rms = float(np.sqrt(np.mean(np.square(original))))
    transformed_rms = float(np.sqrt(np.mean(np.square(transformed))))
    correlation = float(np.corrcoef(original.reshape(-1), transformed.reshape(-1))[0, 1]) if np.std(original) and np.std(transformed) else 1.0
    perturbation_rms = float(np.sqrt(np.mean(np.square(difference))))
    return {
        "group": group,
        "subject": int(subject),
        "file": path.name,
        "original_rms": original_rms,
        "transformed_rms": transformed_rms,
        "rms_ratio": transformed_rms / max(original_rms, 1e-30),
        "perturbation_rms": perturbation_rms,
        "correlation": correlation,
    }


def prepare(input_root: str | Path, output_root: str | Path, condition: str, seed: int = 20260905) -> dict[str, Any]:
    source = Path(input_root).resolve()
    output = Path(output_root).resolve()
    if condition not in SUPPORTED_CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    noise_snr_db = NOISE_CONDITIONS.get(condition)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    groups = {group: _subjects(source, group) for group in ("source", "target", "retention")}
    source_rms = {subject: _subject_rms(source, "source", subject) for subject in groups["source"]}
    reference_rms = float(np.median(list(source_rms.values())))
    subject_rms: dict[tuple[str, int], float] = {}
    for group, subjects in groups.items():
        for subject in subjects:
            subject_rms[(group, subject)] = _subject_rms(source, group, subject)
    records: list[dict[str, Any]] = []
    for group, subjects in groups.items():
        for subject in subjects:
            if condition == "rms_equalized":
                scale = reference_rms / max(subject_rms[(group, subject)], 1e-30)
            else:
                scale = 1.0
            for path in _files(source, group, subject):
                output_data = output / group / str(subject) / "data" / path.name
                record = _copy_or_transform(path, output_data, group, subject, condition, scale, seed, float(noise_snr_db or 0.0))
                label_path = source / group / str(subject) / "label" / path.name
                output_label = output / group / str(subject) / "label" / path.name
                output_label.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(label_path, output_label)
                record["scale"] = float(scale)
                records.append(record)
    perturbation: dict[str, Any] | None
    if noise_snr_db is not None:
        perturbation = {
            "type": "band_limited_additive_noise",
            "snr_db": float(noise_snr_db),
            "low_hz": NOISE_LOW_HZ,
            "high_hz": NOISE_HIGH_HZ,
            "target_only": True,
        }
    elif condition in CROSSTALK_CONDITIONS:
        perturbation = {
            "type": "adjacent_channel_cross_talk",
            "fraction": float(CROSSTALK_CONDITIONS[condition]),
            "target_only": True,
            "matrix": _channel_mixing_matrix(8, CROSSTALK_CONDITIONS[condition]).tolist(),
        }
    elif condition in BASELINE_DRIFT_CONDITIONS:
        perturbation = {
            "type": "low_frequency_baseline_drift",
            "fraction_of_channel_rms": float(BASELINE_DRIFT_CONDITIONS[condition]),
            "frequency_hz": BASELINE_DRIFT_FREQUENCY_HZ,
            "target_only": True,
        }
    elif condition == "target_gain_drift10":
        perturbation = {
            "type": "smooth_gain_drift",
            "fraction": GAIN_DRIFT_FRACTION,
            "period_seconds": GAIN_DRIFT_PERIOD_SECONDS,
            "target_only": True,
        }
    elif condition == "rms_equalized":
        perturbation = {
            "type": "subject_rms_calibration",
            "reference": "source_subject_median_rms",
            "target_only": False,
        }
    else:  # pragma: no cover - guarded by SUPPORTED_CONDITIONS
        perturbation = None
    manifest = {
        "schema_version": 1,
        "dataset": "ISRUC-medium-derived",
        "condition": condition,
        "input_root": str(source),
        "output_root": str(output),
        "seed": int(seed),
        "groups": groups,
        "source_subject_rms": {str(k): v for k, v in source_rms.items()},
        "reference_source_median_rms": reference_rms,
        "noise": ({"snr_db": noise_snr_db, "low_hz": NOISE_LOW_HZ, "high_hz": NOISE_HIGH_HZ} if noise_snr_db is not None else None),
        "perturbation": perturbation,
        "records": records,
        "labels_unchanged": True,
        "scientific_conclusion_allowed": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "DATASET_CARD.md").write_text(
        "# ISRUC medium derived condition\n\n"
        f"Condition: `{condition}`. This directory is derived from the local ISRUC medium subset; labels and epoch boundaries are unchanged. "
        "The transformation is a controlled data-quality/domain-shift experiment, not a replacement dataset. "
        "Cross-talk and baseline-drift conditions affect target epochs only; their exact dose is recorded in `manifest.json`. "
        "See `manifest.json` for per-file RMS, correlation and perturbation diagnostics.\n",
        encoding="utf-8",
    )
    rms_ratios = np.asarray([item["rms_ratio"] for item in records], dtype=np.float64)
    correlations = np.asarray([item["correlation"] for item in records], dtype=np.float64)
    return {
        "condition": condition,
        "files": len(records),
        "rms_ratio_min": float(rms_ratios.min()),
        "rms_ratio_max": float(rms_ratios.max()),
        "correlation_min": float(correlations.min()),
        "correlation_median": float(np.median(correlations)),
        "reference_source_median_rms": reference_rms,
        "scientific_conclusion_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--condition", choices=tuple(sorted(SUPPORTED_CONDITIONS)), required=True)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input_root, args.output_root, args.condition, args.seed), sort_keys=True))


if __name__ == "__main__":
    main()
