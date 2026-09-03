#!/usr/bin/env python3
"""Create a tiny, reproducible ISRUC split outside the repository.

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_for_subject(source: Path, subject: int, count: int) -> list[tuple[Path, Path]]:
    data_dir = source / str(subject) / "data"
    label_dir = source / str(subject) / "label"
    files = sorted(data_dir.glob("*.npy"), key=lambda item: int(item.stem))
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
    parser.add_argument("--target-subject", type=int, default=2)
    parser.add_argument("--retention-subject", type=int, default=5)
    parser.add_argument("--files-per-subject", type=int, default=1)
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
        "target": [args.target_subject],
        "retention": [args.retention_subject],
    }
    records: list[dict[str, object]] = []
    for group, subjects in groups.items():
        for subject in subjects:
            for data_path, label_path in files_for_subject(source, subject, args.files_per_subject):
                destination_data = output / group / str(subject) / "data" / data_path.name
                destination_label = output / group / str(subject) / "label" / label_path.name
                destination_data.parent.mkdir(parents=True, exist_ok=True)
                destination_label.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(data_path, destination_data)
                shutil.copy2(label_path, destination_label)
                records.append(
                    {
                        "group": group,
                        "subject": subject,
                        "file": data_path.stem,
                        "source_data": str(data_path),
                        "source_label": str(label_path),
                        "output_data": str(destination_data),
                        "output_label": str(destination_label),
                        "source_data_sha256": sha256(data_path),
                        "source_label_sha256": sha256(label_path),
                        "output_data_sha256": sha256(destination_data),
                        "output_label_sha256": sha256(destination_label),
                    }
                )
    manifest = {
        "schema": "edgeforge.eeg-mini-split.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "ISRUC group1 processed npy float32",
        "source_root": str(source),
        "output_root": str(output),
        "shape_contract": {"data": [20, 8, 3000], "label": [20]},
        "groups": groups,
        "files_per_subject": args.files_per_subject,
        "records": records,
        "public_release": False,
        "release_note": "Human EEG payload; keep local unless license, ethics and de-identification permit publication.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output_root": str(output), "records": len(records), "manifest": str(output / "manifest.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
