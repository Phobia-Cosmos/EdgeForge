#!/usr/bin/env python3
"""Generate a deterministic EEG-shaped synthetic LoP positive-control dataset.

The data are a controlled benchmark, not a replacement for ISRUC.  Each epoch
has shape ``(8, 3000)`` and contains 16 latent channel/half-window features.
Source labels use feature 0; the ordered target subjects use orthogonal feature
axes.  The runner can then inject an explicit warm-model plasticity lesion to
verify that the LoP gate detects a known positive control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SOURCE_SUBJECTS = (1, 2, 3)
TARGET_SUBJECTS = (4, 5, 6, 7, 8, 9, 10, 11)
RETENTION_SUBJECTS = (12, 13)
TARGET_AXES = (15, 1, 14, 2, 13, 3, 12, 4)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_subject(root: Path, group: str, subject: int, axis: int, samples: int, seed: int, noise: float) -> dict:
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((samples, 16), dtype=np.float32)
    labels = (latent[:, axis] > 0.0).astype(np.int64)
    values = latent.reshape(samples, 8, 2, 1).repeat(1500, axis=3)
    # Add EEG-like temporal structure without changing the latent segment mean.
    time = np.linspace(0.0, 2.0 * np.pi, 1500, dtype=np.float32)
    carrier = 0.08 * np.sin(time)[None, None, None, :]
    channel_phase = np.linspace(0.0, 1.0, 8, dtype=np.float32)[None, :, None, None]
    values = values + carrier * (1.0 + channel_phase)
    values = values + rng.normal(0.0, noise, size=values.shape).astype(np.float32)
    values = values.reshape(samples, 8, 3000)
    data_dir = root / group / str(subject) / "data"
    label_dir = root / group / str(subject) / "label"
    data_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "000.npy"
    label_path = label_dir / "000.npy"
    np.save(data_path, values.astype(np.float32, copy=False), allow_pickle=False)
    np.save(label_path, labels, allow_pickle=False)
    return {
        "group": group,
        "subject": int(subject),
        "axis": int(axis),
        "samples": int(samples),
        "data": str(data_path),
        "label": str(label_path),
        "data_sha256": _sha256(data_path),
        "label_sha256": _sha256(label_path),
        "shape": [int(item) for item in values.shape],
    }


def generate(output: str | Path, *, seed: int = 20260904, source_samples: int = 256, target_samples: int = 128, retention_samples: int = 128, noise: float = 0.35) -> dict:
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for index, subject in enumerate(SOURCE_SUBJECTS):
        records.append(_write_subject(root, "source", subject, 0, source_samples, seed + index * 101, noise))
    for index, (subject, axis) in enumerate(zip(TARGET_SUBJECTS, TARGET_AXES)):
        records.append(_write_subject(root, "target", subject, axis, target_samples, seed + 10_000 + index * 101, noise))
    for index, subject in enumerate(RETENTION_SUBJECTS):
        records.append(_write_subject(root, "retention", subject, 0, retention_samples, seed + 20_000 + index * 101, noise))
    manifest = {
        "schema_version": 1,
        "dataset": "edgeforge-synthetic-eeg-lop-positive-control-v1",
        "seed": int(seed),
        "input_shape": [8, 3000],
        "classes": 2,
        "latent_features": 16,
        "latent_layout": "8 channels × 2 temporal half-windows",
        "source_subjects": list(SOURCE_SUBJECTS),
        "target_subject_order": list(TARGET_SUBJECTS),
        "target_axes": list(TARGET_AXES),
        "retention_subjects": list(RETENTION_SUBJECTS),
        "source_samples_per_subject": int(source_samples),
        "target_samples_per_subject": int(target_samples),
        "retention_samples_per_subject": int(retention_samples),
        "noise_std": float(noise),
        "records": records,
        "scientific_conclusion_allowed": False,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--source-samples", type=int, default=256)
    parser.add_argument("--target-samples", type=int, default=128)
    parser.add_argument("--retention-samples", type=int, default=128)
    parser.add_argument("--noise", type=float, default=0.35)
    args = parser.parse_args()
    manifest = generate(args.output, seed=args.seed, source_samples=args.source_samples, target_samples=args.target_samples, retention_samples=args.retention_samples, noise=args.noise)
    print(json.dumps({"status": "ok", "dataset": manifest["dataset"], "records": len(manifest["records"]), "output": str(args.output.resolve()), "scientific_conclusion_allowed": False}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
