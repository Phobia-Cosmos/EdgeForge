#!/usr/bin/env python3
"""Build and audit a continuous EEG LoP result in one local command.

This command derives a standard trajectory catalog from the continuous runner
summary and evaluates each architecture independently.  It never uploads
results and always keeps ``scientific_conclusion_allowed=false``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

from edgeforge.lop_analysis import evaluate_lop_gate
from edgeforge.lop_audit import load_catalog_evidence


def _load_builder():
    path = Path(__file__).with_name("build-eeg-continuous-trajectory-catalog.py")
    spec = importlib.util.spec_from_file_location("edgeforge_eeg_continuous_catalog", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load catalog adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Continuous EEG LoP gate",
        "",
        f"- Status: `{result['status']}`",
        f"- Outcome: `{result['outcome']}`",
        f"- Seeds: {result['seed_count']}/{result['minimum_seeds']}",
        f"- Transitions: {result['stage_count']}/{result['minimum_stages']}",
        f"- Design consistent: `{result['design_consistent']}`",
        f"- Scientific conclusion allowed: `{result['scientific_conclusion_allowed']}`",
        "",
        "| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in result.get("stage_summaries", []):
        ci = row.get("bootstrap_ci95")
        ci_text = "n/a" if ci is None else f"[{ci[0]:.8g}, {ci[1]:.8g}]"
        lines.append(
            f"| `{row['transition']}` | {row['mean'] if row['mean'] is not None else 'n/a'} | "
            f"{row['seed_count']} | {row['positive_seed_count']} | "
            f"{row['negative_seed_count']} | {row['zero_seed_count']} | {ci_text} |"
        )
    lines.extend(["", "## Reasons", ""])
    for reason in result.get("reasons") or ["all gate checks passed; this remains a candidate only"]:
        lines.append(f"- {reason}")
    lines.extend(["", "Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.", ""])
    return "\n".join(lines)


def audit(
    summary: str | Path,
    output_dir: str | Path,
    *,
    budget: int = 50,
    architectures: list[str] | None = None,
    bootstrap_repeats: int = 2000,
    bootstrap_seed: int = 20260830,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    builder = _load_builder()
    catalog_path = output / "trajectory-catalog.json"
    catalog = builder.build_catalog(summary, catalog_path, budget=budget, architectures=architectures)
    loaded = load_catalog_evidence(catalog_path)
    selected = [str(item.get("method")) for item in catalog.get("experiments", []) if item.get("method")]
    methods = sorted(set(selected))
    if architectures:
        methods = [item for item in methods if item in set(architectures)]
    gate_dir = output / "gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict[str, Any]] = {}
    transitions = [f"{index}->{index + 1}" for index in range(len(loaded["experiments"][0].get("task_order", [])))] if loaded["experiments"] else []
    for method in methods:
        method_experiments = [item for item in loaded["experiments"] if str(item.get("method")) == method]
        method_ids = {str(item.get("experiment_id")) for item in method_experiments}
        result = evaluate_lop_gate(
            method_experiments,
            {key: value for key, value in loaded["metrics_by_experiment"].items() if key in method_ids},
            required_transitions=transitions,
            bootstrap_repeats=bootstrap_repeats,
            bootstrap_seed=bootstrap_seed,
        )
        (gate_dir / f"{method}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (gate_dir / f"{method}.md").write_text(_markdown(result), encoding="utf-8")
        reports[method] = {
            "status": result["status"],
            "seed_count": result["seed_count"],
            "stage_count": result["stage_count"],
            "positive_stage_count": result["positive_stage_count"],
            "direction_supported": result["direction_supported"],
            "design_consistent": result["design_consistent"],
            "reasons": result["reasons"],
            "gate_digest": result["gate_digest"],
        }
    manifest = {
        "schema_version": 1,
        "summary": str(Path(summary).resolve()),
        "catalog": str(catalog_path),
        "budget": int(budget),
        "architectures": methods,
        "reports": reports,
        "scientific_conclusion_allowed": False,
    }
    (output / "audit-summary.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--architecture", action="append", dest="architectures", default=[])
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260830)
    args = parser.parse_args()
    result = audit(
        args.summary,
        args.output_dir,
        budget=args.budget,
        architectures=args.architectures or None,
        bootstrap_repeats=args.bootstrap_repeats,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps({"status": "ok", **result}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
