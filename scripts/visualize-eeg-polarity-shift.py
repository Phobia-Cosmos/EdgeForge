#!/usr/bin/env python3
"""Visualize a label-preserving EEG channel-polarity derivative."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


FS_HZ = 100.0


def _load_first(root: Path, subject: int) -> np.ndarray:
    paths = sorted((root / "target" / str(subject) / "data").glob("*.npy"), key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem)
    if not paths:
        raise FileNotFoundError(f"no target data for subject {subject} under {root}")
    return np.load(paths[0], allow_pickle=False).astype(np.float32, copy=False)


def visualize(clean_root: str | Path, shift_root: str | Path, output_dir: str | Path, subject: int, channel: int) -> dict:
    clean = _load_first(Path(clean_root).resolve(), int(subject))
    shifted = _load_first(Path(shift_root).resolve(), int(subject))
    if clean.shape != shifted.shape or clean.ndim != 3:
        raise ValueError(f"clean/shift shapes do not match: {clean.shape} versus {shifted.shape}")
    if not 0 <= int(channel) < clean.shape[1]:
        raise ValueError(f"channel {channel} is outside {clean.shape[1]} channels")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    samples = min(1000, clean.shape[-1])
    time = np.arange(samples) / FS_HZ
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    axes[0].plot(time, clean[0, channel, :samples], label="clean", linewidth=1.0)
    axes[0].plot(time, shifted[0, channel, :samples], label="polarity-shifted", linewidth=1.0, alpha=0.8)
    axes[0].set(title=f"Subject {subject}, channel {channel}: time-domain polarity inversion", xlabel="time (s)", ylabel="amplitude")
    axes[0].legend()
    frequency, clean_psd = welch(clean[:, channel, :], fs=FS_HZ, nperseg=1024, axis=-1)
    _, shifted_psd = welch(shifted[:, channel, :], fs=FS_HZ, nperseg=1024, axis=-1)
    clean_mean = clean_psd.mean(axis=0)
    shifted_mean = shifted_psd.mean(axis=0)
    mask = (frequency >= 0.5) & (frequency <= 40.0)
    axes[1].plot(frequency[mask], 10.0 * np.log10(clean_mean[mask] + 1e-30), label="clean", linewidth=1.2)
    axes[1].plot(frequency[mask], 10.0 * np.log10(shifted_mean[mask] + 1e-30), label="polarity-shifted", linestyle="--")
    axes[1].set(title="Power spectrum remains unchanged by sign inversion", xlabel="frequency (Hz)", ylabel="power (dB)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "eeg-polarity-waveform-and-psd.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    clean_flat = clean.transpose(1, 0, 2).reshape(clean.shape[1], -1)
    shifted_flat = shifted.transpose(1, 0, 2).reshape(shifted.shape[1], -1)
    clean_corr = np.corrcoef(clean_flat)
    shifted_corr = np.corrcoef(shifted_flat)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, matrix, title, limit in (
        (axes[0], clean_corr, "clean channel correlation", 1.0),
        (axes[1], shifted_corr, "polarity-shifted correlation", 1.0),
        (axes[2], shifted_corr - clean_corr, "shift minus clean", 2.0),
    ):
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set(title=title, xlabel="channel", ylabel="channel")
        fig.colorbar(image, ax=axis, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output / "eeg-polarity-channel-correlation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    selected_corr = float(np.corrcoef(clean_flat[channel], shifted_flat[channel])[0, 1])
    clean_rms = float(np.sqrt(np.mean(np.square(clean_flat[channel].astype(np.float64)))))
    shifted_rms = float(np.sqrt(np.mean(np.square(shifted_flat[channel].astype(np.float64)))))
    psd_relative_error = float(np.linalg.norm(shifted_mean - clean_mean) / max(np.linalg.norm(clean_mean), 1e-30))
    summary = {
        "analysis": "eeg-channel-polarity-paired-visualization-v1",
        "subject": int(subject),
        "channel": int(channel),
        "selected_channel_correlation": selected_corr,
        "selected_channel_rms_ratio": shifted_rms / max(clean_rms, 1e-30),
        "selected_channel_psd_relative_error": psd_relative_error,
        "figures": ["eeg-polarity-waveform-and-psd.png", "eeg-polarity-channel-correlation.png"],
        "scientific_conclusion_allowed": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--shift-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subject", type=int, default=2)
    parser.add_argument("--channel", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(visualize(args.clean_root, args.shift_root, args.output_dir, args.subject, args.channel), sort_keys=True))


if __name__ == "__main__":
    main()
