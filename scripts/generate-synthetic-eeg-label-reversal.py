#!/usr/bin/env python3
"""Create a synthetic EEG target-label-reversal shift.

The source and target waveforms are unchanged.  Only target labels are
recomputed from the opposite polarity of the source feature-0 axis.  This is
an isolated data-shift control for LoP experiments, not a claim about ISRUC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def generate(source_root: str | Path, output_root: str | Path) -> dict:
    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    import shutil

    shutil.copytree(source, output)
    transformed = 0
    for data_path in sorted((output / "target").glob("*/data/*.npy")):
        subject = data_path.parent.parent.name
        values = np.load(data_path, allow_pickle=False).astype(np.float32, copy=False)
        if values.ndim != 3 or tuple(values.shape[1:]) != (8, 3000):
            raise ValueError(f"invalid EEG-shaped target file: {data_path}")
        labels = (values[:, 0, :1500].mean(axis=1) < 0.0).astype(np.int64)
        np.save(output / "target" / subject / "label" / data_path.name, labels, allow_pickle=False)
        transformed += 1
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset"] = "edgeforge-synthetic-eeg-label-reversal-v1"
    manifest["target_shift"] = "all target labels reverse the source feature-0 polarity"
    manifest["scientific_conclusion_allowed"] = False
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"dataset": manifest["dataset"], "transformed_target_files": transformed, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.source_root, args.output_root), sort_keys=True))


if __name__ == "__main__":
    main()
