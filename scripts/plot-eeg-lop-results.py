#!/usr/bin/env python3
"""Render versioned EEG LoP calibration artifacts into small PNG figures.

The script consumes the archived JSON artifacts under ``results/`` by default.
It never reads raw EEG or checkpoints and always records that the figures are
descriptive calibration outputs, not a scientific LoP conclusion.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / "eeg-lop-full" / "v0.19.1"


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _run_paths(run_root: Path) -> list[Path]:
    paths: list[Path] = []
    for architecture_dir in sorted(run_root.iterdir()):
        if not architecture_dir.is_dir():
            continue
        nested = architecture_dir / architecture_dir.name
        paths.extend(sorted(nested.glob("seed*/run.json")))
    return paths


def _save(fig: plt.Figure, output: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(output.name)


def _plot_fresh_gap(analysis: dict[str, Any], output_dir: Path) -> str:
    by_arch: dict[str, list[dict[str, Any]]] = {}
    for row in analysis["architecture_budget_statistics"]:
        by_arch.setdefault(str(row["architecture"]), []).append(row)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for architecture, rows in sorted(by_arch.items()):
        rows = sorted(rows, key=lambda row: int(row["budget"]))
        x = [int(row["budget"]) for row in rows]
        y = [float(row["mean_fresh_gap"]) for row in rows]
        ci_low = [float(row["two_way_seed_subject_cluster_bootstrap_ci95"][0]) for row in rows]
        ci_high = [float(row["two_way_seed_subject_cluster_bootstrap_ci95"][1]) for row in rows]
        (line,) = ax.plot(x, y, marker="o", linewidth=1.8, label=architecture)
        ax.fill_between(x, ci_low, ci_high, color=line.get_color(), alpha=0.12)
    ax.axhline(0.0, color="black", linewidth=0.9, linestyle="--")
    ax.set_xlabel("Adaptation budget (optimizer steps)")
    ax.set_ylabel("Fresh gap (fresh accuracy - warm accuracy)")
    ax.set_title("ISRUC v0.19.1 calibration fresh gap")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, frameon=False)
    return _save(fig, output_dir / "fresh-gap-by-budget.png")


def _plot_source_accuracy(analysis: dict[str, Any], run_paths: list[Path], output_dir: Path) -> str:
    by_arch: dict[str, list[float]] = {}
    for path in run_paths:
        run = _load(path)
        by_arch.setdefault(str(run["architecture"]), []).append(float(run["source"]["accuracy"]))
    architectures = sorted(by_arch)
    means = [statistics.mean(by_arch[name]) for name in architectures]
    errors = [statistics.pstdev(by_arch[name]) for name in architectures]
    majority = float(analysis["acceptance"]["source_majority_accuracy"])

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    positions = list(range(len(architectures)))
    ax.bar(positions, means, yerr=errors, capsize=4, color="#4c78a8", alpha=0.88)
    ax.axhline(majority, color="#d62728", linewidth=1.2, linestyle="--", label=f"majority={majority:.3f}")
    ax.axhline(majority + 0.10, color="#2ca02c", linewidth=1.2, linestyle=":", label=f"gate={majority + 0.10:.3f}")
    ax.set_xticks(positions, architectures)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Held-out source accuracy")
    ax.set_title("ISRUC v0.19.1 source training adequacy")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    return _save(fig, output_dir / "source-accuracy-by-architecture.png")


def _plot_target_curves(run_paths: list[Path], output_dir: Path) -> str:
    selected = {"brainuicl", "tcn", "transformer"}
    curves: dict[str, dict[int, dict[str, list[float]]]] = {}
    for path in run_paths:
        run = _load(path)
        architecture = str(run["architecture"])
        if architecture not in selected:
            continue
        for stage in run["stages"]:
            for side in ("warm", "fresh"):
                for point in stage[side]["curve"]:
                    curves.setdefault(architecture, {}).setdefault(int(point["step"]), {}).setdefault(side, []).append(float(point["accuracy"]))

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for architecture in sorted(curves):
        steps = sorted(curves[architecture])
        color = None
        for side, linestyle in (("warm", "-"), ("fresh", "--")):
            means = [statistics.mean(curves[architecture][step][side]) for step in steps]
            (line,) = ax.plot(steps, means, marker="o", linewidth=1.8, linestyle=linestyle, label=f"{architecture} / {side}")
            color = line.get_color()
    ax.set_xlabel("Adaptation budget (optimizer steps)")
    ax.set_ylabel("Held-out target accuracy")
    ax.set_title("ISRUC v0.19.1 calibration target learning curves")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, frameon=False)
    return _save(fig, output_dir / "target-learning-curves.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    result_root = args.result_root.resolve()
    output_dir = (args.output_dir or result_root / "plots").resolve()
    analysis_path = result_root / "analysis" / "calibration-v2" / "analysis.json"
    run_root = result_root / "runs" / "calibration" / "random-a"
    analysis = _load(analysis_path)
    run_paths = _run_paths(run_root)
    if len(run_paths) != 15:
        raise SystemExit(f"expected 15 calibration run files, found {len(run_paths)} under {run_root}")

    figures = [
        _plot_fresh_gap(analysis, output_dir),
        _plot_source_accuracy(analysis, run_paths, output_dir),
        _plot_target_curves(run_paths, output_dir),
    ]
    manifest = {
        "schema": "edgeforge.eeg-lop-plot-manifest.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_version": analysis.get("experiment_version"),
        "phase": analysis.get("phase"),
        "source_analysis": str(analysis_path),
        "source_run_count": len(run_paths),
        "figures": figures,
        "interpretation": "descriptive calibration visualization; not a scientific LoP conclusion",
        "scientific_conclusion_allowed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plot-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output_dir": str(output_dir), "figures": figures, "scientific_conclusion_allowed": False}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
