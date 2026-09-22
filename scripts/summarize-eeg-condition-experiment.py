#!/usr/bin/env python3
"""Compare raw and label-preserving EEG LoP conditions.

The comparison is deliberately descriptive: it pairs the same TCN stage/seed
cells across conditions, reports fresh-gap changes and reads the condition
gate/utility artifacts.  It never changes data or upgrades a mixed-direction
result into a LoP claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _runs(path: str | Path, architecture: str) -> dict[tuple[int, int, int], float]:
    summary = _load(path)
    rows: dict[tuple[int, int, int], float] = {}
    for run in summary.get("runs", []):
        if run.get("architecture") != architecture:
            continue
        seed = int(run["seed"])
        for stage_index, stage in enumerate(run.get("stages", [])):
            for gap in stage.get("gaps", []):
                rows[(seed, stage_index, int(gap["step"]))] = float(gap["fresh_gap"])
    if not rows:
        raise ValueError(f"no {architecture} rows in {path}")
    return rows


def _summary(path: str | Path, architecture: str) -> dict[str, Any]:
    data = _load(path)
    runs = [run for run in data.get("runs", []) if run.get("architecture") == architecture]
    budgets = sorted({int(gap["step"]) for run in runs for stage in run.get("stages", []) for gap in stage.get("gaps", [])})
    result: dict[str, Any] = {"summary": str(Path(path).resolve()), "seeds": sorted(int(run["seed"]) for run in runs), "source_accuracy": [float(run["source"]["accuracy"]) for run in runs], "budgets": budgets, "budget_metrics": {}}
    for budget in budgets:
        values = [
            float(gap["fresh_gap"])
            for run in runs
            for stage in run.get("stages", [])
            for gap in stage.get("gaps", [])
            if int(gap["step"]) == budget
        ]
        result["budget_metrics"][str(budget)] = {"mean": sum(values) / len(values), "positive": sum(value > 1e-8 for value in values), "zero": sum(abs(value) <= 1e-8 for value in values), "negative": sum(value < -1e-8 for value in values), "count": len(values), "min": min(values), "max": max(values)}
    return result


def _gate(summary_path: Path, budget: int, architecture: str) -> dict[str, Any] | None:
    path = summary_path.parent / "audit-budgets" / f"budget-{budget}" / "gate" / f"{architecture}.json"
    if not path.is_file():
        return None
    data = _load(path)
    return {"status": data.get("status"), "positive_stage_count": data.get("positive_stage_count"), "stage_count": data.get("stage_count"), "direction_supported": data.get("direction_supported"), "path": str(path)}


def compare(baseline: str | Path, conditions: dict[str, str | Path], output_dir: str | Path, architecture: str = "tcn") -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_paths = {"raw": baseline, **conditions}
    stats = {name: _summary(path, architecture) for name, path in all_paths.items()}
    baseline_rows = _runs(baseline, architecture)
    paired: dict[str, Any] = {}
    for name, path in conditions.items():
        rows = _runs(path, architecture)
        deltas: dict[str, Any] = {}
        for budget in sorted({key[2] for key in baseline_rows} & {key[2] for key in rows}):
            values = [rows[key] - baseline_rows[key] for key in baseline_rows if key in rows and key[2] == budget]
            deltas[str(budget)] = {"mean_delta": sum(values) / len(values), "positive_delta": sum(value > 1e-8 for value in values), "negative_delta": sum(value < -1e-8 for value in values), "count": len(values), "min_delta": min(values), "max_delta": max(values)}
        paired[name] = deltas
    gates = {}
    for name, path in all_paths.items():
        summary_path = Path(path).resolve()
        gates[name] = {str(budget): _gate(summary_path, budget, architecture) for budget in (10, 25, 50)}
    result = {"schema_version": 1, "analysis": "eeg-condition-lop-comparison-v1", "architecture": architecture, "conditions": {name: str(Path(path).resolve()) for name, path in all_paths.items()}, "stats": stats, "paired_deltas_vs_raw": paired, "gates": gates, "scientific_conclusion_allowed": False, "interpretation": "descriptive paired comparison; a positive mean fresh-gap or condition delta is not sufficient for a LoP claim"}
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# EEG condition LoP comparison", "", f"This is a paired descriptive comparison of the same `{architecture}` seeds and subject transitions. Positive fresh-gap is the LoP outcome direction, but mixed cells remain inconclusive.", "", "## Aggregate fresh-gap", "", "| condition | budget | mean | positive | zero | negative | min | max | gate |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for name, item in stats.items():
        for budget, values in item["budget_metrics"].items():
            gate = gates[name].get(budget)
            status = "n/a" if gate is None else str(gate["status"])
            lines.append(f"| {name} | {budget} | {values['mean']:+.4f} | {values['positive']} | {values['zero']} | {values['negative']} | {values['min']:+.4f} | {values['max']:+.4f} | `{status}` |")
    lines.extend(["", "## Paired change versus raw", "", "| condition | budget | mean fresh-gap change | cells improved | cells worsened |", "| --- | ---: | ---: | ---: | ---: |"])
    for name, by_budget in paired.items():
        for budget, values in by_budget.items():
            lines.append(f"| {name} | {budget} | {values['mean_delta']:+.4f} | {values['positive_delta']} | {values['negative_delta']} |")
    lines.extend(["", "## Utility interpretation", "", "All compared conditions preserve the same subject order, labels and epoch boundaries. Refer to each derived condition manifest and quality-audit report for the exact waveform-correlation and spectral checks. No condition in this comparison passed the strict all-transition/all-seed gate, so the results should be interpreted as transfer/preprocessing sensitivity rather than induced natural LoP.", ""])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result


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
    result = compare(args.baseline, conditions, args.output_dir, architecture=args.architecture)
    print(json.dumps({"status": "ok", "output_dir": str(Path(args.output_dir).resolve()), "conditions": list(result["conditions"]), "scientific_conclusion_allowed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
