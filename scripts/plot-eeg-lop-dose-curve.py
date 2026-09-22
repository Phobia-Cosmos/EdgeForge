#!/usr/bin/env python3
"""Plot paired EEG LoP fresh-gap dose curves.

The input summaries come from the continuous EEG runner.  This script keeps
the seed/transition cells separate, reports a descriptive normal-approximation
95% interval, and never upgrades a mixed-direction result into a LoP claim.
It writes figures and JSON outside the source checkout when invoked by the
experiment protocol.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SNR_PATTERN = re.compile(r"^target_snr(?P<snr>[0-9]+(?:\.[0-9]+)?)_noise$")
CROSSTALK_PATTERN = re.compile(r"^target_crosstalk(?P<fraction>[0-9]+(?:\.[0-9]+)?)$")
BASELINE_PATTERN = re.compile(r"^target_baseline_drift(?P<fraction>[0-9]+(?:\.[0-9]+)?)$")


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _values(path: str | Path, architecture: str) -> dict[int, list[float]]:
    summary = _load(path)
    values: dict[int, list[float]] = {}
    for run in summary.get("runs", []):
        if str(run.get("architecture")) != architecture:
            continue
        for stage in run.get("stages", []):
            for gap in stage.get("gaps", []):
                step = int(gap["step"])
                values.setdefault(step, []).append(float(gap["fresh_gap"]))
    if not values:
        raise ValueError(f"no {architecture} fresh-gap rows in {path}")
    return values


def _condition_dose(name: str) -> tuple[str, float] | None:
    if name == "raw":
        return "noise_rms_ratio", 0.0
    match = SNR_PATTERN.match(name)
    if match:
        snr_db = float(match.group("snr"))
        return "noise_rms_ratio", float(10.0 ** (-snr_db / 20.0))
    match = CROSSTALK_PATTERN.match(name)
    if match:
        return "cross_talk_fraction", float(match.group("fraction")) / 100.0
    match = BASELINE_PATTERN.match(name)
    if match:
        return "baseline_drift_fraction", float(match.group("fraction")) / 100.0
    if name == "target_gain_drift10":
        return "gain_drift_fraction", 0.10
    return None


def _stats(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if len(array) > 1:
        standard_error = float(array.std(ddof=1) / math.sqrt(len(array)))
    else:
        standard_error = 0.0
    return {
        "count": int(len(array)),
        "mean": mean,
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "standard_error": standard_error,
        "ci95": [mean - 1.96 * standard_error, mean + 1.96 * standard_error],
        "positive": int(np.count_nonzero(array > 1e-8)),
        "zero": int(np.count_nonzero(np.abs(array) <= 1e-8)),
        "negative": int(np.count_nonzero(array < -1e-8)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _gate(summary_path: str | Path, budget: int, architecture: str) -> str | None:
    path = Path(summary_path).resolve().parent / "audit-budgets" / f"budget-{budget}" / "gate" / f"{architecture}.json"
    if not path.is_file():
        return None
    return _load(path).get("status")


def build_curve(
    baseline: str | Path,
    conditions: dict[str, str | Path],
    output_dir: str | Path,
    *,
    architecture: str = "tcn",
) -> dict[str, Any]:
    """Build dose rows and figures from a clean baseline plus conditions."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str | Path] = {"raw": baseline, **conditions}
    loaded = {name: _values(path, architecture) for name, path in paths.items()}
    budgets = sorted({budget for values in loaded.values() for budget in values})
    rows: list[dict[str, Any]] = []
    for condition, by_budget in loaded.items():
        dose = _condition_dose(condition)
        for budget in budgets:
            if budget not in by_budget:
                continue
            row = {
                "condition": condition,
                "budget": int(budget),
                "dose": None if dose is None else float(dose[1]),
                "dose_axis": None if dose is None else dose[0],
                "gate": _gate(paths[condition], budget, architecture),
                "summary": str(Path(paths[condition]).resolve()),
            }
            row.update(_stats(by_budget[budget]))
            rows.append(row)

    result = {
        "schema_version": 1,
        "analysis": "eeg-lop-dose-curve-v1",
        "architecture": architecture,
        "conditions": {name: str(Path(path).resolve()) for name, path in paths.items()},
        "budgets": budgets,
        "rows": rows,
        "excluded_from_shared_axes": [name for name in paths if _condition_dose(name) is None],
        "interval": "normal-approximation-95-percent-over-seed-transition-cells",
        "scientific_conclusion_allowed": False,
        "interpretation": "descriptive dose response; positive mean fresh-gap is not sufficient for a LoP claim",
    }
    (output / "dose-curve.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _plot(result, output)
    lines = [
        "# EEG LoP dose curves",
        "",
        "The curves summarize fresh-gap across the same seed/transition cells. Error bars are descriptive normal-approximation 95% intervals, not a causal or population-level confidence claim.",
        "",
        "| condition | dose axis | dose | budget | mean | 95% CI | positive | zero | negative | gate |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        ci = row["ci95"]
        ci_text = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
        dose_text = "n/a" if row["dose"] is None else f"{float(row['dose']):.4f}"
        lines.append(
            f"| {row['condition']} | {row['dose_axis'] or 'separate'} | "
            f"{dose_text} | {row['budget']} | "
            f"{row['mean']:+.4f} | {ci_text} | {row['positive']} | {row['zero']} | {row['negative']} | "
            f"`{row['gate'] or 'n/a'}` |"
        )
    lines.extend([
        "",
        "`target_snr*` dose is noise RMS / signal RMS, so lower SNR appears farther right. Cross-talk and baseline-drift conditions use their configured fractions as separate axes. `target_gain_drift10` is plotted on its own gain-drift axis; `rms_equalized` is a calibration condition and is retained in JSON but not placed on a nuisance-dose axis.",
        "",
        "All results remain descriptive and `scientific_conclusion_allowed=false`.",
        "",
    ])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def _plot(result: dict[str, Any], output: Path) -> None:
    rows = result["rows"]
    budgets = [int(item) for item in result["budgets"] if int(item) > 0]
    if not budgets:
        budgets = [int(item) for item in result["budgets"]]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2), squeeze=False)
    panels = [
        (axes[0, 0], "noise_rms_ratio", "Noise dose: RMS(noise) / RMS(signal)"),
        (axes[0, 1], "gain_drift_fraction", "Gain drift dose: envelope fraction"),
        (axes[1, 0], "cross_talk_fraction", "Cross-talk dose: adjacent mixing fraction"),
        (axes[1, 1], "baseline_drift_fraction", "Baseline drift dose: fraction of channel RMS"),
    ]
    for ax, axis_name, title in panels:
        selected = [row for row in rows if row["dose_axis"] == axis_name]
        if axis_name != "noise_rms_ratio":
            selected.extend(
                {**row, "dose": 0.0, "dose_axis": axis_name}
                for row in rows
                if row["condition"] == "raw"
            )
        conditions = []
        for row in selected:
            if row["condition"] not in conditions:
                conditions.append(row["condition"])
        for budget in budgets:
            points = []
            for condition in conditions:
                match = next((row for row in selected if row["condition"] == condition and int(row["budget"]) == budget), None)
                if match is not None:
                    points.append(match)
            if not points:
                continue
            points.sort(key=lambda row: float(row["dose"]))
            ax.errorbar(
                [float(row["dose"]) for row in points],
                [float(row["mean"]) for row in points],
                yerr=[1.96 * float(row["standard_error"]) for row in points],
                marker="o",
                linewidth=1.5,
                capsize=3,
                label=f"budget {budget}",
            )
        ax.set_title(title)
        ax.set_xlabel("dose")
        ax.set_ylabel("mean fresh-gap (fresh accuracy - warm accuracy)")
        ax.grid(alpha=0.25)
        if conditions:
            ax.legend(fontsize=8)
    fig.suptitle("EEG LoP perturbation dose response (descriptive)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output / "eeg-lop-dose-curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--condition", action="append", required=True, help="NAME=SUMMARY_PATH; may be repeated")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--architecture", default="tcn")
    args = parser.parse_args()
    conditions: dict[str, str] = {}
    for item in args.condition:
        if "=" not in item:
            parser.error("--condition must use NAME=SUMMARY_PATH")
        name, path = item.split("=", 1)
        conditions[name] = path
    result = build_curve(args.baseline, conditions, args.output_dir, architecture=args.architecture)
    print(json.dumps({"status": "ok", "output_dir": str(Path(args.output_dir).resolve()), "rows": len(result["rows"]), "scientific_conclusion_allowed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
