#!/usr/bin/env python3
"""Create a small, reproducible ISRUC split outside the repository.

The source tree contains human EEG data and is never modified.  This command
copies only paired data/label files into a user-selected local directory and
writes a manifest with source/output hashes.  The resulting directory is
experiment state, not a public dataset artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_diversity_score(label_path: Path) -> tuple[int, int, float, int]:
    """Prefer files whose two halves and full sequence contain more classes."""
    labels = np.load(label_path, allow_pickle=False).reshape(-1)
    if labels.shape != (20,):
        raise ValueError(f"expected 20 labels in {label_path}, got {labels.shape}")
    counts = np.bincount(labels.astype(np.int64), minlength=5)
    probabilities = counts[counts > 0] / labels.size
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    half_coverage = min(len(np.unique(labels[:10])), len(np.unique(labels[10:])))
    return half_coverage, len(np.unique(labels)), entropy, -int(label_path.stem)


def files_for_subject(source: Path, subject: int, count: int, selection_strategy: str) -> list[tuple[Path, Path]]:
    data_dir = source / str(subject) / "data"
    label_dir = source / str(subject) / "label"
    files = sorted(data_dir.glob("*.npy"), key=lambda item: int(item.stem))
    if selection_strategy == "label-diversity":
        files.sort(key=lambda item: label_diversity_score(label_dir / item.name), reverse=True)
    pairs: list[tuple[Path, Path]] = []
    for data_path in files[:count]:
        label_path = label_dir / data_path.name
        if not label_path.is_file():
            raise FileNotFoundError(f"missing paired label for {data_path}")
        pairs.append((data_path, label_path))
    if len(pairs) != count:
        raise RuntimeError(f"subject {subject} has {len(pairs)} paired files; requested {count}")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-subjects", type=int, nargs="+", default=[1, 3, 4])
    parser.add_argument("--target-subject", "--target-subjects", dest="target_subjects", type=int, nargs="+", default=[2])
    parser.add_argument("--retention-subject", "--retention-subjects", dest="retention_subjects", type=int, nargs="+", default=[5])
    parser.add_argument("--files-per-subject", type=int, default=1)
    parser.add_argument("--selection-strategy", choices=("first", "label-diversity"), default="first")
    parser.add_argument("--public-release", action="store_true", help="write a portable manifest without local absolute paths")
    args = parser.parse_args()
    source = args.source_root.resolve()
    output = args.output_root.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output == source or source in output.parents:
        raise ValueError("output-root must not be inside source-root")
    if args.files_per_subject < 1:
        raise ValueError("files-per-subject must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[int]] = {
        "source": list(dict.fromkeys(args.source_subjects)),
        "target": list(dict.fromkeys(args.target_subjects)),
        "retention": list(dict.fromkeys(args.retention_subjects)),
    }
    assigned = [subject for subjects in groups.values() for subject in subjects]
    if len(assigned) != len(set(assigned)):
        raise ValueError("subjects must not overlap across source, target and retention groups")
    records: list[dict[str, object]] = []
    for group, subjects in groups.items():
        for subject in subjects:
            for data_path, label_path in files_for_subject(source, subject, args.files_per_subject, args.selection_strategy):
                destination_data = output / group / str(subject) / "data" / data_path.name
                destination_label = output / group / str(subject) / "label" / label_path.name
                destination_data.parent.mkdir(parents=True, exist_ok=True)
                destination_label.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(data_path, destination_data)
                shutil.copy2(label_path, destination_label)
                record = {
                        "group": group,
                        "subject": subject,
                        "file": data_path.stem,
                        "source_data": str(data_path.relative_to(source)) if args.public_release else str(data_path),
                        "source_label": str(label_path.relative_to(source)) if args.public_release else str(label_path),
                        "output_data": str(destination_data.relative_to(output)) if args.public_release else str(destination_data),
                        "output_label": str(destination_label.relative_to(output)) if args.public_release else str(destination_label),
                        "source_data_sha256": sha256(data_path),
                        "source_label_sha256": sha256(label_path),
                        "output_data_sha256": sha256(destination_data),
                        "output_label_sha256": sha256(destination_label),
                    }
                records.append(record)
    manifest = {
        "schema": "edgeforge.eeg-mini-split.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "ISRUC group1 processed npy float32",
        "source_url": "https://sleeptight.isr.uc.pt/ISRUC_Sleep/",
        "citation": "Khalighi et al., ISRUC-Sleep, Computer Methods and Programs in Biomedicine 124 (2016), 180-192",
        "source_root": None if args.public_release else str(source),
        "output_root": "." if args.public_release else str(output),
        "shape_contract": {"data": [20, 8, 3000], "label": [20]},
        "groups": groups,
        "files_per_subject": args.files_per_subject,
        "selection_strategy": args.selection_strategy,
        "records": records,
        "public_release": args.public_release,
        "release_note": "Public mini subset authorized for repository publication by the repository owner." if args.public_release else "Local human EEG subset; do not publish without explicit authorization.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output_root": str(output), "records": len(records), "manifest": str(output / "manifest.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
