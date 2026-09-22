#!/usr/bin/env python3
"""Convert continuous EEG architecture runs into an EdgeForge trajectory catalog.

The continuous runner deliberately stores a compact, architecture-oriented
``summary.json``.  The existing EdgeForge LoP gate consumes one trajectory
result per independent seed, with explicit fixed-budget design metadata and
metric roles.  This adapter creates those derived result files without
modifying the raw run artifacts or the source data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
DEFAULT_BUDGET = 50
DEFAULT_PROTOCOL = "eeg-continuous-lop-v1"
DEFAULT_SPLIT = "first10-adapt-last10-heldout"
DEFAULT_COMPARISON_GROUP = "ISRUC-medium-continuous-final50"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _finite_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected numeric value, got {value!r}")
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:  # NaN/Inf
        raise ValueError(f"expected finite numeric value, got {value!r}")
    return result


def _budget_gap(stage: dict[str, Any], budget: int) -> float:
    rows = stage.get("gaps")
    if not isinstance(rows, list):
        raise ValueError("stage is missing gaps[]")
    for row in rows:
        if isinstance(row, dict) and int(row.get("step", -1)) == budget:
            return _finite_number(row.get("fresh_gap"))
    raise ValueError(f"stage {stage.get('stage')} has no fresh_gap at budget {budget}")


def _architecture_filter(architectures: Iterable[str] | None) -> set[str] | None:
    if architectures is None:
        return None
    values = {str(item).strip() for item in architectures if str(item).strip()}
    return values or None


def _experiment_spec(
    run: dict[str, Any],
    *,
    subjects: list[int],
    budget: int,
    learning_rate: float,
    dataset_name: str,
    protocol: str,
    split: str,
    comparison_group: str,
) -> dict[str, Any]:
    architecture = str(run.get("architecture") or "").strip()
    seed = run.get("seed")
    if not architecture or not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("every run must declare a non-empty architecture and integer seed")
    parameter_count = run.get("parameters")
    if not isinstance(parameter_count, int) or isinstance(parameter_count, bool):
        raise ValueError(f"run {architecture}/seed{seed} has no integer parameter count")
    experiment_id = f"eeg-continuous-{architecture}-seed{seed}"
    model_structure = {"name": architecture, "parameter_count": parameter_count}
    dataset = {"name": str(dataset_name)}
    return {
        "experiment_id": experiment_id,
        "workload": "raeeg-lop",
        "protocol": protocol,
        "method": architecture,
        "seed": int(seed),
        "dataset": dataset,
        "model": model_structure,
        "model_structure": model_structure,
        "subject": "continuous-target-stream",
        "split": split,
        "task_order": subjects,
        "probe_budget": int(budget),
        "optimizer": {"name": "Adam", "scope": "target-train-only", "state_persisted": False},
        "learning_rate": float(learning_rate),
        "fresh_warm_protocol": "warm-carried-across-subjects-vs-fresh-reinit",
        "runner": {"mode": "import", "adapter": "eeg-continuous-lop-v1"},
        "metadata": {
            "comparison_group": comparison_group,
            "dataset": dataset,
            "subject": "continuous-target-stream",
            "split": split,
            "task_order": subjects,
            "probe_budget": int(budget),
            "optimizer": {"name": "Adam", "scope": "target-train-only", "state_persisted": False},
            "learning_rate": float(learning_rate),
            "model_structure": model_structure,
            "fresh_warm_protocol": "warm-carried-across-subjects-vs-fresh-reinit",
            "scientific_conclusion_allowed": False,
        },
    }


def _derived_metrics(
    run: dict[str, Any],
    *,
    budget: int,
    target_subjects: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stages = run.get("stages")
    if not isinstance(stages, list) or len(stages) != len(target_subjects):
        raise ValueError(f"run {run.get('architecture')}/seed{run.get('seed')} has an invalid stage count")
    metrics: list[dict[str, Any]] = []
    final_gaps: list[float] = []
    for index, (stage, subject) in enumerate(zip(stages, target_subjects)):
        if not isinstance(stage, dict):
            raise ValueError("stage records must be JSON objects")
        gap = _budget_gap(stage, budget)
        final_gaps.append(gap)
        context = {
            "source_stage": index,
            "target_stage": index + 1,
            "target_subject": int(subject),
            "stage_index": index,
            "probe_budget": int(budget),
            "adapter_protocol": "continuous-final-budget-v1",
            "metric_role": "outcome",
        }
        metrics.append({
            "name": "task.plasticity.fresh_gap",
            "value": gap,
            "step": index + 1,
            "context": context,
        })
        warm = stage.get("warm") if isinstance(stage.get("warm"), dict) else {}
        fresh = stage.get("fresh") if isinstance(stage.get("fresh"), dict) else {}
        warm_final = warm.get("final") if isinstance(warm.get("final"), dict) else {}
        fresh_final = fresh.get("final") if isinstance(fresh.get("final"), dict) else {}
        for prefix, values in (("warm", warm_final), ("fresh", fresh_final)):
            for metric_name, source_name in (
                ("accuracy", "retention_accuracy"),
                ("loss", "retention_loss"),
                ("macro_f1", "retention_macro_f1"),
            ):
                if source_name not in values:
                    continue
                metrics.append({
                    "name": f"task.probe.retention.{prefix}_{metric_name}",
                    "value": _finite_number(values[source_name]),
                    "step": index + 1,
                    "context": {
                        "source_stage": index,
                        "target_stage": index + 1,
                        "target_subject": int(subject),
                        "probe_budget": int(budget),
                        "adapter_protocol": "continuous-final-budget-v1",
                        "metric_role": "retention",
                    },
                })
    summary = {
        "architecture": run.get("architecture"),
        "seed": run.get("seed"),
        "target_subjects": target_subjects,
        "probe_budget": int(budget),
        "final_fresh_gap": final_gaps,
        "scientific_conclusion_allowed": False,
    }
    return metrics, summary


def build_catalog(
    summary_path: str | Path,
    output_path: str | Path,
    *,
    budget: int = DEFAULT_BUDGET,
    architectures: Iterable[str] | None = None,
    protocol: str = DEFAULT_PROTOCOL,
    split: str = DEFAULT_SPLIT,
    comparison_group: str = DEFAULT_COMPARISON_GROUP,
) -> dict[str, Any]:
    summary_file = Path(summary_path).resolve()
    output_file = Path(output_path).resolve()
    if budget <= 0:
        raise ValueError("budget must be positive")
    source = _load(summary_file)
    runs = source.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("summary must contain a non-empty runs[] list")
    metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    target_subjects = [int(item) for item in metadata.get("target_subject_order", [])]
    if not target_subjects:
        raise ValueError("summary metadata must contain target_subject_order")
    learning_rate = _finite_number(metadata.get("adapt_lr"))
    if learning_rate <= 0.0:
        raise ValueError("summary metadata adapt_lr must be positive")
    dataset_name = str(metadata.get("dataset") or source.get("dataset") or "unknown-dataset")
    if comparison_group == DEFAULT_COMPARISON_GROUP and budget != DEFAULT_BUDGET:
        comparison_group = f"ISRUC-medium-continuous-final{budget}"
    allowed = _architecture_filter(architectures)
    selected = [run for run in runs if isinstance(run, dict) and (allowed is None or str(run.get("architecture")) in allowed)]
    if not selected:
        raise ValueError("no runs matched the requested architecture filter")
    seen: set[tuple[str, int]] = set()
    root = output_file.parent
    experiments: list[dict[str, Any]] = []
    output_file.parent.mkdir(parents=True, exist_ok=True)
    for run in sorted(selected, key=lambda item: (str(item.get("architecture")), int(item.get("seed", -1)))):
        architecture = str(run.get("architecture") or "")
        seed = run.get("seed")
        identity = (architecture, int(seed))
        if identity in seen:
            raise ValueError(f"duplicate architecture/seed run: {architecture}/seed{seed}")
        seen.add(identity)
        spec = _experiment_spec(run, subjects=target_subjects, budget=budget, learning_rate=learning_rate, dataset_name=dataset_name, protocol=protocol, split=split, comparison_group=comparison_group)
        metrics, derived_summary = _derived_metrics(run, budget=budget, target_subjects=target_subjects)
        result_relative = Path("trajectories") / architecture / f"seed{seed}.json"
        result_path = root / result_relative
        result_path.parent.mkdir(parents=True, exist_ok=True)
        source_run_path = (summary_file.parent / architecture / f"seed{seed}" / "run.json").resolve()
        result = {
            "schema_version": SCHEMA_VERSION,
            "instrumentation": "edgeforge-eeg-continuous-trajectory-v1",
            "status": "succeeded",
            "experiment_id": spec["experiment_id"],
            "spec": spec,
            "metrics": metrics,
            "summary": derived_summary,
            "source_run": {
                "path": str(source_run_path),
                "sha256": _sha256(source_run_path) if source_run_path.is_file() else None,
                "summary_path": str(summary_file),
                "summary_sha256": _sha256(summary_file),
            },
            "scientific_conclusion_allowed": False,
        }
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        spec["runner"] = {"mode": "import", "adapter": "eeg-continuous-lop-v1", "result_path": str(result_relative)}
        experiments.append(spec)
    catalog = {
        "schema_version": SCHEMA_VERSION,
        "worker_work_root": str(root),
        "source": {
            "name": "EdgeForge continuous EEG trajectory adapter",
            "summary_path": str(summary_file),
            "summary_sha256": _sha256(summary_file),
            "result_count": len(experiments),
            "probe_budget": int(budget),
        },
        "defaults": {},
        "experiments": experiments,
        "scientific_conclusion_allowed": False,
    }
    output_file.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--architecture", action="append", dest="architectures", default=[])
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--comparison-group", default=DEFAULT_COMPARISON_GROUP)
    args = parser.parse_args()
    catalog = build_catalog(
        args.summary,
        args.output,
        budget=args.budget,
        architectures=args.architectures or None,
        protocol=args.protocol,
        split=args.split,
        comparison_group=args.comparison_group,
    )
    print(json.dumps({
        "status": "ok",
        "experiments": len(catalog["experiments"]),
        "catalog": str(args.output.resolve()),
        "budget": args.budget,
        "scientific_conclusion_allowed": False,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
