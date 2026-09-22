#!/usr/bin/env python3
"""Select EEG data transformations from measured LoP-related diagnostics.

This controller is deliberately an experiment planner, not an unbounded
optimizer. It ranks label-preserving nuisance conditions against a baseline
using quality, fresh-gap and representation-state evidence, then emits a
reproducible next-run plan. Raw data are never modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CONDITIONS = {
    "clean": {"dataset_condition": "clean", "quality_risk": 0.0, "mechanism": "baseline"},
    "rms_equalized": {"dataset_condition": "rms_equalized", "quality_risk": 0.0, "mechanism": "acquisition_gain_equalization"},
    "target_crosstalk10": {"dataset_condition": "target_crosstalk10", "quality_risk": 0.20, "mechanism": "cross_channel_mixing"},
    "target_channel_polarity": {"dataset_condition": "target_channel_polarity", "quality_risk": 0.15, "mechanism": "signed_lead_change"},
    "target_snr15_noise": {"dataset_condition": "target_snr15_noise", "quality_risk": 0.35, "mechanism": "band_limited_noise"},
    "target_snr10_noise": {"dataset_condition": "target_snr10_noise", "quality_risk": 0.55, "mechanism": "band_limited_noise"},
    "target_baseline_drift10": {"dataset_condition": "target_baseline_drift10", "quality_risk": 0.35, "mechanism": "low_frequency_drift"},
    "target_gain_drift10": {"dataset_condition": "target_gain_drift10", "quality_risk": 0.10, "mechanism": "smooth_gain"},
    "target_montage_swap": {"dataset_condition": "target_montage_swap", "quality_risk": 0.25, "mechanism": "signed_permutation"},
}


def _rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for run in summary.get("runs", []):
        for stage in run.get("stages", []):
            for gap in stage.get("gaps", []):
                rows.append({"architecture": run.get("architecture"), "seed": run.get("seed"), "subject": stage.get("subject"), **gap})
    return rows


def _condition_score(summary: dict[str, Any], quality: dict[str, Any] | None, *, budget: int, architecture: str) -> dict[str, Any]:
    rows = [row for row in _rows(summary) if row.get("architecture") == architecture and int(row.get("step", -1)) == budget]
    gaps = [float(row["fresh_gap"]) for row in rows if row.get("fresh_gap") is not None]
    positives = sum(value > 0.0 for value in gaps)
    negatives = sum(value < 0.0 for value in gaps)
    quality_ok = bool(quality and quality.get("preferred_waveform_gate_pass"))
    # If the paired quality audit is unavailable, mark the result unknown
    # rather than silently treating it as a failed signal-quality condition.
    quality_status = "pass" if quality_ok else ("fail" if quality is not None else "unknown")
    mean_gap = sum(gaps) / len(gaps) if gaps else None
    # A condition is useful only when it preserves the signal and moves the
    # warm/fresh outcome consistently. Mixed signs are explicitly penalized.
    direction = (positives - negatives) / len(gaps) if gaps else -1.0
    consistency = positives / len(gaps) if gaps else 0.0
    quality_penalty = 0.0 if quality_ok else (0.5 if quality is None else 1.0)
    score = (0.50 * consistency) + (0.25 * max(0.0, float(mean_gap or 0.0))) + (0.25 * direction) - 0.50 * quality_penalty
    return {"budget": budget, "architecture": architecture, "cells": len(gaps), "mean_fresh_gap": mean_gap, "positive_cells": positives, "negative_cells": negatives, "zero_cells": len(gaps) - positives - negatives, "direction_balance": direction, "quality_gate_pass": quality_ok, "quality_status": quality_status, "score": score}


def plan(
    baseline: str | Path,
    conditions: dict[str, str | Path],
    output_dir: str | Path,
    *,
    architecture: str = "tcn",
    budget: int = 25,
    quality_paths: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    base = json.loads(Path(baseline).read_text(encoding="utf-8"))
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name, path in {"clean": baseline, **conditions}.items():
        summary = json.loads(Path(path).read_text(encoding="utf-8"))
        quality_path = Path(quality_paths[name]).resolve() if quality_paths and name in quality_paths else Path(path).parent / "quality-audit.json"
        quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.is_file() else None
        info = CONDITIONS.get(name, {"dataset_condition": name, "quality_risk": None, "mechanism": "unknown"})
        score = _condition_score(summary, quality, budget=budget, architecture=architecture)
        records.append({"condition": name, **info, **score, "summary": str(Path(path).resolve()), "quality_summary": (str(quality_path.resolve()) if quality_path.is_file() else None)})
    records.sort(key=lambda row: float(row["score"]), reverse=True)
    ranked = [row for row in records if row["condition"] != "clean"]
    selected = ranked[:2]
    result = {
        "schema_version": 1,
        "analysis": "metric-driven-eeg-lop-induction-plan",
        "architecture": architecture,
        "budget": budget,
        "baseline": str(Path(baseline).resolve()),
        "ranked_conditions": records,
        "next_conditions": [row["condition"] for row in selected],
        "next_experiment": {
            "keep_fixed": ["source training recipe", "target subject order", "three independent seeds", "per-file 10/10 adaptation/evaluation split", "fresh/warm initialization", "fixed budgets"],
            "quality_gate": ["labels byte-identical", "shape/finite pass", "correlation >= 0.98 where applicable", "no clipping", "PSD and RMS changes reported"],
            "lop_gate": "positive fresh-gap for every transition and seed with positive seed-cluster bootstrap lower bound",
            "mechanism_gate": ["activation coverage or near-zero state changes before gap", "gradient nonzero fraction remains measurable", "rank/spectral change is reproducible", "reset/ReDo counterfactual reduces fresh-gap"],
        },
        "scientific_conclusion_allowed": False,
    }
    (output / "plan.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Metric-driven EEG LoP induction plan", "", f"Architecture: `{architecture}`; budget: `{budget}`", "", "| condition | score | mean fresh-gap | +/0/- | quality | mechanism |", "| --- | ---: | ---: | --- | --- | --- |"]
    for row in records:
        lines.append(f"| `{row['condition']}` | {row['score']:.4f} | {row['mean_fresh_gap'] if row['mean_fresh_gap'] is not None else 'n/a'} | {row['positive_cells']}/{row['zero_cells']}/{row['negative_cells']} | `{row['quality_status']}` | {row['mechanism']} |")
    lines.extend(["", f"Next conditions: {', '.join(f'`{name}`' for name in result['next_conditions']) or 'none'}", "", "This ranks candidates; it does not claim LoP. A condition is accepted only after the quality, strict fresh-gap and mechanism counterfactual gates pass.", ""])
    (output / "PLAN.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--condition", action="append", default=[], help="NAME=SUMMARY_JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--architecture", default="tcn")
    parser.add_argument("--budget", type=int, default=25)
    parser.add_argument("--quality", action="append", default=[], help="NAME=QUALITY_AUDIT_JSON")
    args = parser.parse_args()
    conditions = {}
    for item in args.condition:
        name, separator, path = item.partition("=")
        if not separator or not name or not path:
            parser.error("--condition must use NAME=SUMMARY_JSON")
        conditions[name] = path
    quality_paths = {}
    for item in args.quality:
        name, separator, path = item.partition("=")
        if not separator or not name or not path:
            parser.error("--quality must use NAME=QUALITY_AUDIT_JSON")
        quality_paths[name] = path
    result = plan(args.baseline, conditions, args.output_dir, architecture=args.architecture, budget=args.budget, quality_paths=quality_paths)
    print(json.dumps({"status": "ok", "next_conditions": result["next_conditions"], "scientific_conclusion_allowed": False}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
