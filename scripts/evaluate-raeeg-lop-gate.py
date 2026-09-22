#!/usr/bin/env python3
"""Evaluate the fixed-budget RA-EEG LoP requirement gate locally.

The command only reads a catalog and its result bundles.  A ``candidate``
result means the preregistered evidence checks passed; it is not a scientific
or causal conclusion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from edgeforge.lop_analysis import DEFAULT_LOP_OUTCOME, evaluate_lop_gate
from edgeforge.lop_audit import load_catalog_evidence


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# RA-EEG LoP Requirement Gate",
        "",
        f"- Status: `{result['status']}`",
        f"- Outcome: `{result['outcome']}`",
        f"- Seeds: {result['seed_count']}/{result['minimum_seeds']}",
        f"- Stages/transitions: {result['stage_count']}/{result['minimum_stages']}",
        f"- Scientific conclusion allowed: `{result['scientific_conclusion_allowed']}`",
        "",
        "## Stage Evidence",
        "",
        "| Transition | Seeds | Mean | Median | Std | 95% CI | Positive seeds | Direction |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for item in result.get("stage_summaries", []):
        ci = item.get("bootstrap_ci95")
        ci_text = "n/a" if ci is None else f"[{ci[0]:.8g}, {ci[1]:.8g}]"
        lines.append(
            f"| `{item['transition']}` | {item['seed_count']} | {item['mean'] if item['mean'] is not None else 'n/a'} | "
            f"{item['median'] if item['median'] is not None else 'n/a'} | {item['std'] if item['std'] is not None else 'n/a'} | "
            f"{ci_text} | {item['positive_seed_count']}/{item['seed_count']} | "
            f"`{'supported' if item['direction_supported'] else 'not-supported'}` |"
        )
    lines.extend(["", "## Reasons", ""])
    reasons = result.get("reasons") or ["all gate checks passed; this remains a candidate only"]
    for reason in reasons:
        lines.append(f"- {reason}")
    lines.extend([
        "",
        "Retention/forgetting metrics are inventory only and are never substituted for fresh-gap.",
        "",
        f"Gate digest: `{result['gate_digest']}`",
        "",
    ])
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--outcome", default=DEFAULT_LOP_OUTCOME)
    parser.add_argument("--required-stage", action="append", type=int, default=[])
    parser.add_argument("--required-transition", action="append", default=[], help="transition such as 0->1")
    parser.add_argument("--minimum-seeds", type=int, default=3)
    parser.add_argument("--minimum-stages", type=int, default=2)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260830)
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--output", type=Path, help="write the JSON gate report")
    parser.add_argument("--markdown-output", type=Path, help="write a Markdown gate report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    loaded = load_catalog_evidence(args.catalog, methods=args.method)
    result = evaluate_lop_gate(
        loaded["experiments"],
        loaded["metrics_by_experiment"],
        outcome=args.outcome,
        required_stages=args.required_stage,
        required_transitions=args.required_transition,
        minimum_seeds=args.minimum_seeds,
        minimum_stages=args.minimum_stages,
        bootstrap_repeats=args.bootstrap_repeats,
        bootstrap_seed=args.bootstrap_seed,
    )
    result["catalog"] = loaded["catalog_path"]
    result["catalog_records"] = loaded["records"]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "candidate" else 2


if __name__ == "__main__":
    raise SystemExit(main())
