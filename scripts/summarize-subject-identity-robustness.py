#!/usr/bin/env python3
"""Aggregate multi-seed subject-identity robustness experiments.

The input directories are outputs from ``subject_identity_granularity.py``.
This script never re-fits a decoder: it summarizes known-subject identity
accuracy, threshold-free unknown-subject AUROC, and the k-epoch aggregation
curve across deterministic subject/group splits.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    "known_test_top1",
    "known_test_balanced_top1",
    "unknown_score_auroc",
    "unknown_rejection_rate_at_enrollment_5pct_threshold",
)


def _stats(values: list[float | None]) -> dict[str, Any]:
    finite = np.asarray([float(value) for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    if not len(finite):
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None, "values": []}
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "values": finite.tolist(),
    }


def summarize(paths: list[Path]) -> dict[str, Any]:
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not runs:
        raise ValueError("at least one summary.json is required")
    datasets = sorted({run["dataset"] for run in runs})
    output: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment": "subject-identity-multiseed-robustness-summary-v1",
        "source_files": [str(path.resolve()) for path in paths],
        "datasets": {},
        "scientific_conclusion_allowed": False,
    }
    for dataset in datasets:
        selected = [run for run in runs if run["dataset"] == dataset]
        stages = list(selected[0]["stages"]["sequence"])
        stage_summary: dict[str, Any] = {}
        for stage in stages:
            stage_summary[stage] = {
                metric: _stats([run["stages"]["sequence"][stage].get(metric) for run in selected])
                for metric in METRICS
            }
        curve: dict[str, Any] = {}
        for k in sorted(selected[0].get("aggregation_curve", {}), key=int):
            curve[k] = {
                stage: _stats([run["aggregation_curve"][k][stage]["known_test_top1"] for run in selected])
                for stage in selected[0]["aggregation_curve"][k]
            }
        output["datasets"][dataset] = {
            "run_count": len(selected),
            "seeds": [int(Path(path).parent.name.rsplit("seed", 1)[-1]) for path in paths if json.loads(path.read_text(encoding="utf-8"))["dataset"] == dataset],
            "sequence_count": int(selected[0]["sequence_count"]),
            "known_subject_count": len(selected[0]["known_subjects"]),
            "unknown_subject_count": len(selected[0]["unknown_subjects"]),
            "stages": stage_summary,
            "aggregation_curve": curve,
            "interpretation": {
                "primary_closed_set_metric": "known_test_top1",
                "primary_open_set_metric": "unknown_score_auroc",
                "threshold_note": "unknown rejection uses a confidence threshold calibrated from enrollment predictions; interpret it only with AUROC because known-test false rejection is not present in source summaries",
            },
        }
    return output


def _write_plot(summary: dict[str, Any], output: Path) -> None:
    datasets = list(summary["datasets"])
    stages = list(next(iter(summary["datasets"].values()))["stages"])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    x = np.arange(len(stages), dtype=np.float64)
    width = 0.8 / max(1, len(datasets))
    for dataset_index, dataset in enumerate(datasets):
        data = summary["datasets"][dataset]
        offset = (dataset_index - (len(datasets) - 1) / 2.0) * width
        means = [data["stages"][stage]["known_test_top1"]["mean"] for stage in stages]
        stds = [data["stages"][stage]["known_test_top1"]["std"] for stage in stages]
        axes[0].bar(x + offset, means, width=width, yerr=stds, capsize=3, label=dataset)
        auroc = [data["stages"][stage]["unknown_score_auroc"]["mean"] for stage in stages]
        auroc_std = [data["stages"][stage]["unknown_score_auroc"]["std"] for stage in stages]
        axes[1].bar(x + offset, auroc, width=width, yerr=auroc_std, capsize=3, label=dataset)
        curve_stages = list(next(iter(data["aggregation_curve"].values())))
        for stage in curve_stages:
            ks = [int(value) for value in data["aggregation_curve"]]
            values = [data["aggregation_curve"][str(k)][stage]["mean"] for k in ks]
            axes[2].plot(ks, values, marker="o", label=f"{dataset}:{stage}")
    axes[0].set_title("Known-subject sequence identity")
    axes[0].set_ylabel("top-1 accuracy (mean ± sample std)")
    axes[0].set_xticks(x, stages, rotation=35, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Unseen-subject detection")
    axes[1].set_ylabel("unknown-score AUROC")
    axes[1].set_xticks(x, stages, rotation=35, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[2].set_title("Identity evidence vs observed epochs")
    axes[2].set_xlabel("epochs aggregated per complete group")
    axes[2].set_ylabel("known-subject top-1")
    axes[2].set_xticks(sorted({int(k) for data in summary["datasets"].values() for k in data["aggregation_curve"]}))
    axes[2].set_ylim(0, 1)
    axes[2].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "subject-identity-robustness.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _write_report(summary: dict[str, Any], output: Path) -> None:
    lines = [
        "# Subject-ID feature robustness across splits",
        "",
        "This report aggregates deterministic subject/group splits. The source checkpoint remains sleep/emotion-supervised; subject ID is decoded by a separate frozen-representation probe. Complete sequences/trials are kept disjoint between enrollment and evaluation.",
        "",
    ]
    for dataset, data in summary["datasets"].items():
        lines.extend([
            f"## {dataset}",
            "",
            f"Runs: {data['run_count']} seeds; {data['sequence_count']:,} complete groups; {data['known_subject_count']} enrolled and {data['unknown_subject_count']} unseen subjects per run.",
            "",
            "| representation | known top-1 mean ± std | unknown AUROC mean ± std |",
            "| --- | ---: | ---: |",
        ])
        for stage, values in data["stages"].items():
            known = values["known_test_top1"]
            auroc = values["unknown_score_auroc"]
            lines.append(f"| {stage} | {_fmt(known['mean'])} ± {_fmt(known['std'])} | {_fmt(auroc['mean'])} ± {_fmt(auroc['std'])} |")
        lines.extend(["", "Epoch aggregation:", "", "| epochs | " + " | ".join(next(iter(data["aggregation_curve"].values()))) + " |", "| ---: | " + " | ".join("---:" for _ in next(iter(data["aggregation_curve"].values()))) + " |"])
        for k, stages in data["aggregation_curve"].items():
            lines.append("| " + k + " | " + " | ".join(_fmt(values["mean"]) for values in stages.values()) + " |")
        lines.append("")
    lines.extend([
        "## Interpretation boundary",
        "",
        "Known-subject top-1 answers whether a held-out complete group can be assigned to an already enrolled identity. Unknown AUROC answers whether confidence tends to separate unseen from enrolled subjects; it does not provide the unseen subject's concrete ID. The raw thresholded rejection rate is secondary because the source runs did not save known-test false-rejection rate.",
        "",
        "These experiments establish repeatable subject information, not biometric uniqueness and not LoP. The next control must hold sleep/emotion class fixed, followed by checkpoint-wise tracking during continual learning.",
    ])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize([path.resolve() for path in args.summaries])
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_plot(summary, output)
    _write_report(summary, output)
    print(json.dumps({"status": "ok", "datasets": list(summary["datasets"]), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
