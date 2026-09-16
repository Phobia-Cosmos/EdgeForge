#!/usr/bin/env python3
"""Aggregate multi-seed class-conditioned subject-ID controls.

The inputs are ``summary.json`` files produced by
``subject_identity_class_control.py``.  The aggregation keeps the class
histogram-only baseline separate from the class-conditioned representation
probe and reports mean/sample-standard-deviation across seeds.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _stats(values: list[float | None]) -> dict[str, Any]:
    finite = np.asarray([float(v) for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None, "values": []}
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "min": float(finite.min()),
        "max": float(finite.max()),
        "values": finite.tolist(),
    }


def summarize(paths: list[Path]) -> dict[str, Any]:
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not runs:
        raise ValueError("no summary files")
    output: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment": "subject-identity-class-conditioned-control-multiseed-summary-v1",
        "source_files": [str(p.resolve()) for p in paths],
        "datasets": {},
        "scientific_conclusion_allowed": False,
    }
    for dataset in sorted({r["dataset"] for r in runs}):
        selected = [r for r in runs if r["dataset"] == dataset]
        stages = list(selected[0]["class_conditioned_macro"])
        macro = {
            stage: {
                "balanced_accuracy": _stats([r["class_conditioned_macro"][stage]["macro_balanced_accuracy_mean"] for r in selected]),
                "class_to_class_std": _stats([r["class_conditioned_macro"][stage]["macro_balanced_accuracy_std"] for r in selected]),
            }
            for stage in stages
        }
        hist = selected[0]["class_histogram_only_sequence_probe"]
        output["datasets"][dataset] = {
            "run_count": len(selected),
            "seeds": [int(r["split_seed"]) for r in selected],
            "sequence_count": int(selected[0]["sequence_count"]),
            "subject_count": int(selected[0]["subject_count"]),
            "class_count": int(selected[0]["class_count"]),
            "class_histogram_only_sequence_probe": {
                key: _stats([r["class_histogram_only_sequence_probe"].get(key) for r in selected])
                for key in ("accuracy_top1", "balanced_accuracy_top1")
            },
            "class_conditioned_macro": macro,
            "chance_note": "Per-class chance is 1/N for the included subjects and is recorded in each source run; macro values are balanced accuracy.",
        }
    return output


def write_report(summary: dict[str, Any], output: Path) -> None:
    lines = [
        "# Subject-ID class-conditioned control (multi-seed)",
        "",
        "This report fixes the task label (sleep stage for ISRUC or emotion class for FACED) and samples equal train/test epoch counts per subject. The checkpoint is frozen and the identity decoder is an external class-balanced Ridge probe.",
        "",
    ]
    for dataset, data in summary["datasets"].items():
        lines += [f"## {dataset}", "", f"Runs: {data['run_count']} split seeds; {data['sequence_count']:,} complete groups; {data['subject_count']} subjects; {data['class_count']} task classes.", "", "Class-histogram-only baseline:", "", "| metric | mean ± sample std |", "| --- | ---: |"]
        for key, value in data["class_histogram_only_sequence_probe"].items():
            lines.append(f"| {key} | {value['mean']:.4f} ± {value['std']:.4f} |")
        lines += ["", "Class-conditioned epoch probe (macro balanced accuracy):", "", "| representation | mean ± sample std |", "| --- | ---: |"]
        for stage, value in data["class_conditioned_macro"].items():
            metric = value["balanced_accuracy"]
            lines.append(f"| {stage} | {metric['mean']:.4f} ± {metric['std']:.4f} |")
        lines += ["", "The histogram baseline tests whether subject identity can be explained by different task-label proportions. The conditioned probe removes that explanation by evaluating within a single task class. Above-chance values still may reflect session, electrode placement, impedance, gain, or acquisition-device cues; they are not biometric-authentication claims and are not LoP outcomes.", ""]
    lines += ["## Next use in continual learning", "", "Apply the same frozen probe to checkpoints before training and after each subject transition. Track identity accuracy/AUROC jointly with task accuracy, effective rank, gradient coverage, and fresh-subject adaptation gap. A change in identity decodability alone does not establish plasticity loss; LoP requires a corresponding degradation in learning on a fresh task or subject.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_plot(summary: dict[str, Any], output: Path) -> None:
    datasets = list(summary["datasets"])
    stages = list(next(iter(summary["datasets"].values()))["class_conditioned_macro"])
    x = np.arange(len(stages), dtype=float)
    width = 0.8 / max(1, len(datasets))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, dataset in enumerate(datasets):
        data = summary["datasets"][dataset]
        offset = (i - (len(datasets) - 1) / 2) * width
        means = [data["class_conditioned_macro"][s]["balanced_accuracy"]["mean"] for s in stages]
        stds = [data["class_conditioned_macro"][s]["balanced_accuracy"]["std"] for s in stages]
        ax.bar(x + offset, means, width=width, yerr=stds, capsize=3, label=dataset)
    ax.set_xticks(x, stages, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("macro balanced subject-ID accuracy")
    ax.set_title("Subject-ID decoding with task class held fixed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "class-conditioned-subject-id-multiseed.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize([p.resolve() for p in args.summaries])
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(summary, output)
    write_plot(summary, output)
    print(json.dumps({"status": "ok", "datasets": list(summary["datasets"]), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
