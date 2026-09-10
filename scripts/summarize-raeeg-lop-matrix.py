#!/usr/bin/env python3
"""Create a compact, descriptive report from a LoP matrix trajectory catalog.

The report is deliberately a presentation aid, not a second statistical
engine.  It reads the immutable bundles referenced by a trajectory catalog,
selects one value per metric/stage/context, and emits JSON plus Markdown.
Missing metrics remain ``null``.  No interpolation, imputation, significance
test or LoP conclusion is performed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


METRICS = {
    "effective_rank": "task.spectra.transformer_1.effective_rank",
    "stable_rank": "task.spectra.transformer_1.stable_rank",
    "fresh_gap": "task.plasticity.fresh_gap",
    "fresh_loss_gap": "task.plasticity.fresh_loss_gap",
    "acc_gain": "plasticity.acc_gain",
    "parameter_relative_update": "task.parameter_updates.global_relative_update",
    "representation_drift": "task.representation.transformer_1.drift_from_previous",
    "unlabeled_entropy": "task.unlabeled.predictive_entropy_normalized_mean",
    "unlabeled_confidence": "task.unlabeled.max_probability_mean",
    "unlabeled_agreement": "task.unlabeled.perturbation_prediction_agreement",
    "unlabeled_representation_cosine": "task.unlabeled.representation_cosine_mean",
    "retention_acc_drop": "task.forgetting.checkpoint_acc_drop",
    "retention_mf1_drop": "task.forgetting.checkpoint_mf1_drop",
    "retention_loss_delta": "task.forgetting.checkpoint_loss_delta",
}

METRIC_ROLES = {
    "effective_rank": "predictor",
    "stable_rank": "predictor",
    "fresh_gap": "outcome",
    "fresh_loss_gap": "outcome",
    "acc_gain": "outcome",
    "parameter_relative_update": "diagnostic",
    "representation_drift": "diagnostic",
    "unlabeled_entropy": "diagnostic",
    "unlabeled_confidence": "diagnostic",
    "unlabeled_agreement": "diagnostic",
    "unlabeled_representation_cosine": "diagnostic",
    "retention_acc_drop": "retention",
    "retention_mf1_drop": "retention",
    "retention_loss_delta": "retention",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _metric_values(bundle: dict[str, Any], name: str) -> dict[int, float]:
    values: dict[int, list[float]] = {}
    for item in bundle.get("metrics") or []:
        if item.get("name") != name or item.get("step") is None:
            continue
        value = _finite(item.get("value"))
        if value is None:
            continue
        values.setdefault(int(item["step"]), []).append(value)
    return {step: sum(items) / len(items) for step, items in values.items()}


def build_report(catalog_path: Path) -> dict[str, Any]:
    catalog_path = catalog_path.resolve()
    catalog = _load(catalog_path)
    root = Path(str(catalog.get("worker_work_root") or catalog_path.parent)).resolve()
    rows: list[dict[str, Any]] = []
    for spec in catalog.get("experiments") or []:
        if not isinstance(spec, dict):
            continue
        runner = spec.get("runner") or {}
        result_path = runner.get("result_path")
        if not isinstance(result_path, str):
            continue
        bundle_path = (root / result_path).resolve()
        if not bundle_path.is_file() or not bundle_path.is_relative_to(root):
            continue
        bundle = _load(bundle_path)
        metadata = spec.get("metadata") or {}
        values = {key: _metric_values(bundle, metric) for key, metric in METRICS.items()}
        stages = sorted({stage for mapping in values.values() for stage in mapping})
        for stage in stages:
            row = {
                "experiment_id": spec.get("experiment_id"),
                "dataset": (spec.get("dataset") or {}).get("name"),
                "condition": (spec.get("dataset") or {}).get("condition", "clean"),
                "method": spec.get("method"),
                "subject": metadata.get("subject"),
                "seed": spec.get("seed"),
                "stage": stage,
                "values": {key: mapping.get(stage) for key, mapping in values.items()},
            }
            rows.append(row)
    rows.sort(key=lambda row: (str(row.get("dataset")), str(row.get("condition")), str(row.get("method")), int(row.get("seed") or 0), int(row["stage"])))
    row_groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (row.get("dataset"), row.get("method"), row.get("subject"), row.get("seed"), row.get("stage"))
        row_groups.setdefault(key, {})[str(row.get("condition") or "clean")] = row
    condition_comparisons: list[dict[str, Any]] = []
    for key, by_condition in sorted(row_groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        clean = by_condition.get("clean")
        if clean is None:
            continue
        for condition, shifted in sorted(by_condition.items()):
            if condition == "clean":
                continue
            deltas: dict[str, float | None] = {}
            for metric in METRICS:
                left = _finite(shifted["values"].get(metric))
                right = _finite(clean["values"].get(metric))
                deltas[metric] = left - right if left is not None and right is not None else None
            condition_comparisons.append({
                "dataset": key[0],
                "method": key[1],
                "subject": key[2],
                "seed": key[3],
                "stage": key[4],
                "clean_condition": "clean",
                "shift_condition": condition,
                "shift_minus_clean": deltas,
            })
    methods = sorted({str(row.get("method")) for row in rows})
    seeds = sorted({row.get("seed") for row in rows if isinstance(row.get("seed"), int)})
    return {
        "schema_version": 1,
        "report": "raeeg-lop-matrix-descriptive-v1",
        "catalog": str(catalog_path),
        "row_count": len(rows),
        "methods": methods,
        "seeds": seeds,
        "scientific_conclusion_allowed": False,
        "interpretation": "descriptive checkpoint trajectory table; not a causal LoP conclusion",
        "metric_definitions": METRICS,
        "metric_roles": METRIC_ROLES,
        "rows": rows,
        "condition_comparisons": condition_comparisons,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# RA-EEG LoP matrix descriptive report",
        "",
        f"- Catalog: `{report['catalog']}`",
        f"- Rows: {report['row_count']}; methods: {', '.join(report['methods']) or 'none'}; seeds: {', '.join(str(item) for item in report['seeds']) or 'none'}",
        "- Scientific conclusion allowed: `false`",
        "- Metric roles: predictor (ER), outcome (fresh-gap/plasticity), retention (old-task stability), diagnostic (mechanism/label-free).",
        "",
        "The table is a descriptive checkpoint trajectory. `fresh_gap` is the fixed-budget fresh-vs-checkpoint accuracy gap; positive values mean the fresh control finished higher in the current probe. Missing values are not imputed.",
        "",
        "| method | condition | stage | effective rank | stable rank | fresh gap | fresh loss gap | acc gain | relative parameter update | representation drift | retention ACC drop | retention MF1 drop |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        values = row["values"]
        def fmt(value: Any) -> str:
            return "—" if value is None else f"{float(value):.6g}"
        lines.append(
            f"| {row.get('method')} | {row.get('condition')} | {row.get('stage')} | {fmt(values.get('effective_rank'))} | {fmt(values.get('stable_rank'))} | {fmt(values.get('fresh_gap'))} | {fmt(values.get('fresh_loss_gap'))} | {fmt(values.get('acc_gain'))} | {fmt(values.get('parameter_relative_update'))} | {fmt(values.get('representation_drift'))} | {fmt(values.get('retention_acc_drop'))} | {fmt(values.get('retention_mf1_drop'))} |"
        )
    if report.get("condition_comparisons"):
        lines.extend([
            "",
            "## Shift minus clean (paired descriptive deltas)",
            "",
            "These deltas compare the same dataset/method/subject/seed/stage. They describe domain-shift sensitivity and are not LoP outcomes.",
            "",
            "| method | shift condition | stage | Δ effective rank | Δ fresh gap | Δ fresh loss gap | Δ acc gain | Δ unlabeled entropy | Δ unlabeled confidence | Δ consistency |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for item in report["condition_comparisons"]:
            values = item["shift_minus_clean"]
            def fmt_delta(value: Any) -> str:
                return "—" if value is None else f"{float(value):+.6g}"
            lines.append(
                f"| {item.get('method')} | {item.get('shift_condition')} | {item.get('stage')} | {fmt_delta(values.get('effective_rank'))} | {fmt_delta(values.get('fresh_gap'))} | {fmt_delta(values.get('fresh_loss_gap'))} | {fmt_delta(values.get('acc_gain'))} | {fmt_delta(values.get('unlabeled_entropy'))} | {fmt_delta(values.get('unlabeled_confidence'))} | {fmt_delta(values.get('unlabeled_agreement'))} |"
            )
    lines.extend(["", "No row by itself establishes LoP. Use the accompanying audit for stage-grid, scope and seed evidence gates.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    report = build_report(args.catalog)
    json_output = args.json_output or args.catalog.with_name("matrix-report.json")
    markdown_output = args.markdown_output or args.catalog.with_name("matrix-report.md")
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"row_count": report["row_count"], "json": str(json_output), "markdown": str(markdown_output), "scientific_conclusion_allowed": False}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
