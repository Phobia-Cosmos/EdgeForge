#!/usr/bin/env python3
"""Create a provenance-preserving, controlled EEG domain-shift derivative.

The canonical ISRUC/FACED datasets are never modified.  This utility writes
only selected transformed ``data/*.npy`` files and a manifest that records
source/output digests, transformation parameters, labels and sampling rate.
It is intended for *paired* LoP stress tests: a shift may make adaptation
harder, but it is not evidence of LoP until a clean-data and fresh-init probe
with the same budget is run as a control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


DATASET = {
    "ISRUC": {"channels": 8, "samples": 3000, "sampling_rate": 100.0},
    "FACED": {"channels": 32, "samples": 2500, "sampling_rate": 250.0},
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def subjects(root: Path, selected: str | None) -> list[tuple[str, Path]]:
    wanted = {item.strip() for item in selected.split(",") if item.strip()} if selected else None
    found: list[tuple[str, Path]] = []
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir() or not (candidate / "data").is_dir():
            continue
        name = candidate.name
        numeric = name.removeprefix("sub-")
        if wanted is not None and name not in wanted and numeric not in wanted:
            continue
        found.append((name, candidate))
    if not found:
        raise ValueError(f"no subject directories found under {root}")
    return found


def _channel_indices(channels: int, fraction: float, seed: int) -> np.ndarray:
    count = max(1, min(channels, int(round(channels * fraction))))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(channels, size=count, replace=False))


def transform(values: np.ndarray, *, shift: str, severity: float, sampling_rate: float, seed: int, band_low: float, band_high: float) -> tuple[np.ndarray, dict[str, object]]:
    """Apply one label-preserving signal transformation."""

    rng = np.random.default_rng(seed)
    output = values.astype(np.float32, copy=True)
    channels = output.shape[-2]
    samples = output.shape[-1]
    metadata: dict[str, object] = {"shift": shift, "severity": float(severity)}
    if shift == "amplitude":
        if severity <= 0:
            raise ValueError("amplitude severity must be positive")
        output *= np.float32(severity)
        metadata["scale"] = float(severity)
    elif shift == "noise":
        if severity < 0:
            raise ValueError("noise severity must be non-negative")
        # Scale noise per channel by the clean standard deviation.  This
        # keeps the parameter dimensionless across ISRUC/FACED normalization.
        channel_std = output.std(axis=(-2, -1), keepdims=True).astype(np.float32)
        output += rng.normal(size=output.shape).astype(np.float32) * channel_std * np.float32(severity)
        metadata["noise_std_relative_to_channel_std"] = float(severity)
    elif shift == "channel-dropout":
        if not 0 < severity <= 1:
            raise ValueError("channel-dropout severity must be in (0, 1]")
        indices = _channel_indices(channels, severity, seed)
        output[:, indices, :] = 0.0
        metadata["dropped_channels"] = indices.tolist()
    elif shift == "time-jitter":
        # A bounded circular shift is label-preserving for an epoch-level
        # classification task and avoids changing array dimensions.
        max_shift = max(1, int(round(abs(severity) * samples)))
        offsets = rng.integers(-max_shift, max_shift + 1, size=output.shape[0])
        for index, offset in enumerate(offsets):
            output[index] = np.roll(output[index], int(offset), axis=-1)
        metadata["max_fraction_of_epoch"] = float(abs(severity))
        metadata["offset_samples"] = offsets.tolist()
    elif shift == "bandstop":
        if not 0 <= band_low < band_high <= sampling_rate / 2:
            raise ValueError("bandstop requires 0 <= band_low < band_high <= Nyquist")
        frequencies = np.fft.rfftfreq(samples, d=1.0 / sampling_rate)
        mask = (frequencies >= band_low) & (frequencies <= band_high)
        spectrum = np.fft.rfft(output, axis=-1)
        spectrum[..., mask] *= np.float32(max(0.0, 1.0 - min(1.0, severity)))
        output = np.fft.irfft(spectrum, n=samples, axis=-1).astype(np.float32)
        metadata.update({"band_low_hz": float(band_low), "band_high_hz": float(band_high), "remaining_band_gain": float(max(0.0, 1.0 - min(1.0, severity)))})
    else:
        raise ValueError(f"unsupported shift: {shift}")
    return output, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subjects", help="comma-separated subject names/IDs; default: all")
    parser.add_argument("--max-files", type=int, default=0, help="limit files per subject; 0 means all")
    parser.add_argument("--shift", choices=("amplitude", "noise", "channel-dropout", "time-jitter", "bandstop"), required=True)
    parser.add_argument("--severity", type=float, required=True)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--band-low", type=float, default=8.0)
    parser.add_argument("--band-high", type=float, default=12.0)
    parser.add_argument("--overwrite", action="store_true", help="allow replacing files inside the explicitly supplied output root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = DATASET[args.dataset]
    source = args.source_root.resolve()
    output = args.output_root.resolve()
    if source == output or output.is_relative_to(source):
        raise SystemExit("output-root must not be the source dataset or a descendant of it")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(f"output-root is non-empty; choose a new path or pass --overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    subject_rows = subjects(source, args.subjects)
    for subject_name, subject_root in subject_rows:
        data_paths = sorted((subject_root / "data").glob("*.npy"), key=lambda item: int(item.stem) if item.stem.isdigit() else item.stem)
        if args.max_files > 0:
            data_paths = data_paths[: args.max_files]
        if not data_paths:
            continue
        target_root = output / subject_name
        (target_root / "data").mkdir(parents=True, exist_ok=True)
        (target_root / "label").mkdir(parents=True, exist_ok=True)
        for index, source_data in enumerate(data_paths):
            source_label = subject_root / "label" / source_data.name
            if not source_label.is_file():
                raise FileNotFoundError(source_label)
            values = np.load(source_data, mmap_mode=None)
            if tuple(values.shape) != (20, spec["channels"], spec["samples"]):
                raise ValueError(f"{source_data} shape {values.shape} does not match {args.dataset} contract")
            transformed, transform_metadata = transform(
                values,
                shift=args.shift,
                severity=args.severity,
                sampling_rate=float(spec["sampling_rate"]),
                seed=args.seed + index,
                band_low=args.band_low,
                band_high=args.band_high,
            )
            target_data = target_root / "data" / source_data.name
            target_label = target_root / "label" / source_data.name
            np.save(target_data, transformed, allow_pickle=False)
            # Labels are immutable source references in the manifest; a small
            # symlink keeps the derived tree from duplicating target data.
            if target_label.exists() or target_label.is_symlink():
                target_label.unlink()
            target_label.symlink_to(source_label)
            records.append({
                "subject": subject_name,
                "data": str(target_data),
                "label": str(source_label),
                "source_data": str(source_data),
                "source_data_sha256": digest(source_data),
                "output_data_sha256": digest(target_data),
                "label_sha256": digest(source_label),
                "transform": transform_metadata,
            })
    if not records:
        raise SystemExit("no files transformed")
    manifest = {
        "schema_version": 1,
        "manifest": "raeeg-controlled-shift-v1",
        "dataset": args.dataset,
        "source_root": str(source),
        "output_root": str(output),
        "sampling_rate_hz": spec["sampling_rate"],
        "shape_contract": [20, spec["channels"], spec["samples"]],
        "label_policy": "source-label-symlink; labels unchanged",
        "scientific_conclusion_allowed": False,
        "control_requirement": "pair with clean data, fresh initialization, fixed-budget probe and equal task order",
        "config": {key: getattr(args, key) for key in ("shift", "severity", "seed", "band_low", "band_high", "max_files")},
        "records": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": args.dataset, "shift": args.shift, "records": len(records), "manifest": str(output / "manifest.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
