#!/usr/bin/env python3
"""Summarize checkpoint diagnostics emitted by the continuous EEG runner.

The report joins representation rank, activation, gradient and parameter
probes with the fixed-budget fresh-gap.  It is a descriptive mechanism report:
none of these diagnostics replaces the multi-seed LoP gate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


METRICS = (
    "embedding_effective_rank",
    "embedding_effective_rank_normalized",
    "block1_effective_rank",
    "gradient_norm_l2",
    "gradient_nonzero_fraction",
    "parameter_global_relative_update",
    "classifier_input_near_zero_fraction",
)


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if float(x.std()) <= 0.0 or float(y.std()) <= 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _diagnostic_row(
    run: dict[str, Any],
    stage: dict[str, Any],
    arm: str,
    curve_row: dict[str, Any],
    gap_by_step: dict[int, float],
) -> dict[str, Any] | None:
    diagnostic = curve_row.get("diagnostics")
    if not isinstance(diagnostic, dict) or diagnostic.get("status") != "computed":
        return None
    representations = diagnostic.get("representations", {})
    embedding = representations.get("embedding", {})
    block1 = representations.get("block1", {})
    classifier_input = representations.get("classifier_input", {})
    gradient = diagnostic.get("gradient", {})
    parameter_norm = diagnostic.get("parameter_norm", {})
    return {
        "seed": int(run["seed"]),
        "stage": int(stage["stage"]),
        "subject": int(stage["subject"]),
        "arm": arm,
        "budget": int(curve_row["step"]),
        "fresh_gap": gap_by_step.get(int(curve_row["step"])),
        "embedding_effective_rank": embedding.get("effective_rank"),
        "embedding_effective_rank_normalized": embedding.get("effective_rank_normalized"),
        "block1_effective_rank": block1.get("effective_rank"),
        "gradient_norm_l2": gradient.get("norm_l2"),
        "gradient_nonzero_fraction": gradient.get("nonzero_fraction"),
        "parameter_global_relative_update": parameter_norm.get("global_relative_update"),
        "classifier_input_near_zero_fraction": classifier_input.get("near_zero_fraction"),
    }


def _mean_stats(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    values = [float(row[metric]) for row in rows if row.get(metric) is not None and np.isfinite(float(row[metric]))]
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def summarize(summary: str | Path, output_dir: str | Path, *, architecture: str = "tcn") -> dict[str, Any]:
    data = _load(summary)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for run in data.get("runs", []):
        if str(run.get("architecture")) != architecture:
            continue
        for stage in run.get("stages", []):
            gap_by_step = {int(gap["step"]): float(gap["fresh_gap"]) for gap in stage.get("gaps", [])}
            for arm in ("warm", "fresh"):
                for curve_row in stage.get(arm, {}).get("curve", []):
                    row = _diagnostic_row(run, stage, arm, curve_row, gap_by_step)
                    if row is not None:
                        rows.append(row)
    if not rows:
        raise ValueError(f"no checkpoint diagnostics for {architecture} in {summary}")
    budgets = sorted({int(row["budget"]) for row in rows})
    arms = sorted({str(row["arm"]) for row in rows})
    aggregate: dict[str, Any] = {}
    for arm in arms:
        for budget in budgets:
            selected = [row for row in rows if row["arm"] == arm and int(row["budget"]) == budget]
            aggregate[f"{arm}:{budget}"] = {
                "arm": arm,
                "budget": int(budget),
                "rows": len(selected),
                "metrics": {metric: _mean_stats(selected, metric) for metric in METRICS},
            }
    initial_rank: list[float] = []
    final_gap: list[float] = []
    max_budget = max(budgets)
    for row in rows:
        if row["arm"] == "warm" and int(row["budget"]) == 0 and row.get("embedding_effective_rank") is not None:
            match = next(
                (
                    candidate
                    for candidate in rows
                    if candidate["seed"] == row["seed"]
                    and candidate["stage"] == row["stage"]
                    and candidate["arm"] == "warm"
                    and int(candidate["budget"]) == max_budget
                ),
                None,
            )
            if match is not None and match.get("fresh_gap") is not None:
                initial_rank.append(float(row["embedding_effective_rank"]))
                final_gap.append(float(match["fresh_gap"]))
    result = {
        "schema_version": 1,
        "analysis": "eeg-checkpoint-diagnostics-summary-v1",
        "summary": str(Path(summary).resolve()),
        "architecture": architecture,
        "budgets": budgets,
        "arms": arms,
        "row_count": len(rows),
        "rows": rows,
        "aggregate": aggregate,
        "lagged": {
            "warm_initial_embedding_rank_to_final_budget_fresh_gap_pearson": _pearson(initial_rank, final_gap),
            "pair_count": len(initial_rank),
            "interpretation": "exploratory correlation across stage-seed cells; not causal evidence",
        },
        "scientific_conclusion_allowed": False,
    }
    (output / "diagnostic-summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# EEG checkpoint diagnostics",
        "",
        "These summaries join representation geometry, activation sparsity, gradient magnitude and parameter movement with the fixed-budget fresh-gap. They are descriptive mechanism diagnostics; the LoP gate remains the required outcome test.",
        "",
        "| arm | budget | rows | embedding ER | embedding ER norm | block1 ER | gradient norm | gradient nonzero | parameter relative update | classifier near-zero |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in sorted(aggregate, key=lambda item: (item.split(":")[0], int(item.split(":")[1]))):
        item = aggregate[key]
        metrics = item["metrics"]
        values = [metrics[name]["mean"] for name in METRICS]
        formatted = ["n/a" if value is None else f"{value:.6g}" for value in values]
        lines.append(
            f"| {item['arm']} | {item['budget']} | {item['rows']} | "
            f"{formatted[0]} | {formatted[1]} | {formatted[2]} | {formatted[3]} | "
            f"{formatted[4]} | {formatted[5]} | {formatted[6]} |"
        )
    lagged = result["lagged"]
    correlation = lagged["warm_initial_embedding_rank_to_final_budget_fresh_gap_pearson"]
    correlation_text = "n/a" if correlation is None else f"{correlation:.6f}"
    lines.extend([
        "",
        f"Warm initial embedding rank versus final-budget fresh-gap Pearson: `{correlation_text}` over {lagged['pair_count']} stage-seed pairs.",
        "",
        "A rank, gradient or activation trend is not sufficient to label LoP; it must be paired with a stable multi-seed, multi-transition fresh-gap direction and retention checks.",
        "",
    ])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--architecture", default="tcn")
    args = parser.parse_args()
    result = summarize(args.summary, args.output_dir, architecture=args.architecture)
    print(json.dumps({"status": "ok", "output_dir": str(Path(args.output_dir).resolve()), "rows": result["row_count"], "scientific_conclusion_allowed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
