#!/usr/bin/env python3
"""Compare EEG adaptation strategies under one matched LoP protocol.

The report pairs identical seed/subject/budget cells across plain adaptation,
source replay, L2-SP and their combination.  Retention is reported separately
from the fresh-gap outcome, and mixed-direction results are never promoted to
a LoP claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DESIGN_KEYS = (
    "dataset",
    "source_subjects",
    "target_subject_order",
    "retention_subjects",
    "target_split",
    "input_shape",
    "classes",
    "source_epochs",
    "source_lr",
    "budgets",
    "adapt_lr",
    "batch_size",
    "fresh_mode",
    "warm_mode",
)


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _strategy(summary: dict[str, Any]) -> str:
    return str(summary.get("adaptation_strategy") or summary.get("metadata", {}).get("adaptation_strategy") or "plain")


def _fingerprint(summary: dict[str, Any]) -> dict[str, Any]:
    metadata = summary.get("metadata", {})
    return {key: metadata.get(key) for key in DESIGN_KEYS}


def _cells(summary: dict[str, Any], architecture: str) -> dict[tuple[int, int, int, int], dict[str, float]]:
    rows: dict[tuple[int, int, int, int], dict[str, float]] = {}
    for run in summary.get("runs", []):
        if run.get("architecture") != architecture:
            continue
        seed = int(run["seed"])
        for stage_index, stage in enumerate(run.get("stages", [])):
            warm_by_step = {int(item["step"]): item for item in stage["warm"]["curve"]}
            fresh_by_step = {int(item["step"]): item for item in stage["fresh"]["curve"]}
            for gap in stage.get("gaps", []):
                budget = int(gap["step"])
                warm = warm_by_step[budget]
                fresh = fresh_by_step[budget]
                rows[(seed, stage_index, int(stage["subject"]), budget)] = {
                    "fresh_gap": float(gap["fresh_gap"]),
                    "warm_accuracy": float(warm["accuracy"]),
                    "fresh_accuracy": float(fresh["accuracy"]),
                    "warm_retention_accuracy": float(warm["retention_accuracy"]),
                    "fresh_retention_accuracy": float(fresh["retention_accuracy"]),
                }
    if not rows:
        raise ValueError(f"no {architecture} cells found")
    return rows


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _gate(summary_path: Path, budget: int, architecture: str) -> dict[str, Any] | None:
    path = summary_path.parent / "audit-budgets" / f"budget-{budget}" / "gate" / f"{architecture}.json"
    if not path.is_file():
        return None
    payload = _load(path)
    return {
        "status": payload.get("status"),
        "direction_supported": payload.get("direction_supported"),
        "positive_stage_count": payload.get("positive_stage_count"),
        "stage_count": payload.get("stage_count"),
        "path": str(path.resolve()),
    }


def compare(
    strategies: dict[str, str | Path],
    output_dir: str | Path,
    *,
    architecture: str = "tcn",
    budgets: tuple[int, ...] = (25, 50),
) -> dict[str, Any]:
    if "plain" not in strategies:
        raise ValueError("a plain=SUMMARY_PATH baseline is required")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    payloads = {name: _load(path) for name, path in strategies.items()}
    for name, payload in payloads.items():
        observed = _strategy(payload)
        if observed != name:
            raise ValueError(f"strategy name mismatch for {name}: summary records {observed}")
    reference = _fingerprint(payloads["plain"])
    mismatches = {
        name: {key: {"plain": reference[key], "strategy": fingerprint[key]} for key in DESIGN_KEYS if reference[key] != fingerprint[key]}
        for name, payload in payloads.items()
        if (fingerprint := _fingerprint(payload)) != reference
    }
    mismatches = {name: values for name, values in mismatches.items() if values}
    if mismatches:
        raise ValueError(f"strategy summaries do not share one protocol: {mismatches}")
    cells = {name: _cells(payload, architecture) for name, payload in payloads.items()}
    reference_keys = set(cells["plain"])
    for name, rows in cells.items():
        if set(rows) != reference_keys:
            raise ValueError(f"strategy {name} has a different seed/stage/budget grid")

    stats: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    for name, rows in cells.items():
        stats[name] = {}
        gates[name] = {}
        for budget in budgets:
            selected = [value for key, value in rows.items() if key[3] == budget]
            if not selected:
                raise ValueError(f"budget {budget} missing for {name}")
            gaps = [item["fresh_gap"] for item in selected]
            stats[name][str(budget)] = {
                "fresh_gap_mean": _mean(gaps),
                "positive": sum(value > 1e-8 for value in gaps),
                "zero": sum(abs(value) <= 1e-8 for value in gaps),
                "negative": sum(value < -1e-8 for value in gaps),
                "count": len(gaps),
                "minimum": min(gaps),
                "maximum": max(gaps),
                "warm_accuracy_mean": _mean([item["warm_accuracy"] for item in selected]),
                "fresh_accuracy_mean": _mean([item["fresh_accuracy"] for item in selected]),
                "warm_retention_accuracy_mean": _mean([item["warm_retention_accuracy"] for item in selected]),
                "fresh_retention_accuracy_mean": _mean([item["fresh_retention_accuracy"] for item in selected]),
            }
            gates[name][str(budget)] = _gate(Path(strategies[name]).resolve(), budget, architecture)
        if name != "plain":
            paired[name] = {}
            for budget in budgets:
                deltas = [
                    rows[key]["fresh_gap"] - cells["plain"][key]["fresh_gap"]
                    for key in sorted(reference_keys)
                    if key[3] == budget
                ]
                retention_deltas = [
                    rows[key]["warm_retention_accuracy"] - cells["plain"][key]["warm_retention_accuracy"]
                    for key in sorted(reference_keys)
                    if key[3] == budget
                ]
                paired[name][str(budget)] = {
                    "fresh_gap_mean_delta": _mean(deltas),
                    "fresh_gap_improved_cells": sum(value > 1e-8 for value in deltas),
                    "fresh_gap_worsened_cells": sum(value < -1e-8 for value in deltas),
                    "warm_retention_accuracy_mean_delta": _mean(retention_deltas),
                    "count": len(deltas),
                }

    result = {
        "schema_version": 1,
        "analysis": "eeg-adaptation-strategy-lop-comparison-v1",
        "architecture": architecture,
        "budgets": [int(item) for item in budgets],
        "strategies": {name: str(Path(path).resolve()) for name, path in strategies.items()},
        "design_fingerprint": reference,
        "stats": stats,
        "paired_deltas_vs_plain": paired,
        "gates": gates,
        "scientific_conclusion_allowed": False,
        "interpretation": "fresh-gap is the LoP outcome; retention remains separate and mixed-direction cells block a LoP claim",
    }
    (output / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# EEG adaptation strategy LoP comparison",
        "",
        f"Matched `{architecture}` comparison. Positive fresh-gap is the LoP direction; a positive mean is insufficient when any seed/transition cell is zero or negative.",
        "",
        "| strategy | budget | fresh-gap mean | + / 0 / - | warm accuracy | warm retention | gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, by_budget in stats.items():
        for budget, values in by_budget.items():
            gate = gates[name][budget]
            status = "n/a" if gate is None else str(gate["status"])
            lines.append(
                f"| {name} | {budget} | {values['fresh_gap_mean']:+.4f} | "
                f"{values['positive']} / {values['zero']} / {values['negative']} | "
                f"{values['warm_accuracy_mean']:.4f} | {values['warm_retention_accuracy_mean']:.4f} | `{status}` |"
            )
    lines.extend([
        "",
        "| strategy | budget | fresh-gap Δ vs plain | improved / worsened cells | warm-retention Δ |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for name, by_budget in paired.items():
        for budget, values in by_budget.items():
            lines.append(
                f"| {name} | {budget} | {values['fresh_gap_mean_delta']:+.4f} | "
                f"{values['fresh_gap_improved_cells']} / {values['fresh_gap_worsened_cells']} | "
                f"{values['warm_retention_accuracy_mean_delta']:+.4f} |"
            )
    lines.extend([
        "",
        "Retention is reported as a separate stability inventory and cannot satisfy the LoP gate. All conclusions remain developmental (`scientific_conclusion_allowed=false`).",
        "",
    ])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", action="append", required=True, help="NAME=SUMMARY_PATH; include plain")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--architecture", default="tcn")
    parser.add_argument("--budgets", nargs="+", type=int, default=[25, 50])
    args = parser.parse_args()
    strategies: dict[str, str] = {}
    for item in args.strategy:
        if "=" not in item:
            parser.error("--strategy must use NAME=SUMMARY_PATH")
        name, path = item.split("=", 1)
        strategies[name] = path
    result = compare(strategies, args.output_dir, architecture=args.architecture, budgets=tuple(args.budgets))
    print(json.dumps({"status": "ok", "output_dir": str(Path(args.output_dir).resolve()), "strategies": list(result["strategies"]), "scientific_conclusion_allowed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
