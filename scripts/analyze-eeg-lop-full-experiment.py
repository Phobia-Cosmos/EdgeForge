#!/usr/bin/env python3
"""Audit and summarize a preregistered full continuous EEG LoP experiment.

The analyzer accepts one or more ``ORDER=summary.json`` inputs, or discovers
those summaries from a full-experiment run manifest.  Missing and
learning-inadequate cells remain visible in the audit, but only valid cells
contribute to descriptive statistics.  Passing every gate produces candidate
evidence only; this command never turns one experiment into a scientific
conclusion automatically.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA = "edgeforge.eeg-lop-full-analysis.v1"
BOOTSTRAP_METHOD = "two-way-seed-subject-cluster-percentile"


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(item) for item in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items)


def _index_rows(rows: Any, *, field: str, context: str) -> tuple[dict[int, dict[str, Any]], list[str]]:
    indexed: dict[int, dict[str, Any]] = {}
    reasons: list[str] = []
    if not isinstance(rows, list):
        return indexed, [f"{context}_{field}_not_list"]
    for row in rows:
        if not isinstance(row, dict) or isinstance(row.get("step"), bool):
            reasons.append(f"{context}_{field}_invalid_row")
            continue
        try:
            step = int(row["step"])
        except (KeyError, TypeError, ValueError):
            reasons.append(f"{context}_{field}_invalid_step")
            continue
        if step in indexed:
            reasons.append(f"{context}_{field}_duplicate_step_{step}")
            continue
        indexed[step] = row
    return indexed, reasons


def source_majority_accuracy(plan: dict[str, Any]) -> float:
    roles = plan.get("roles")
    profiles = plan.get("profiles")
    if not isinstance(roles, dict) or not isinstance(roles.get("source"), list) or not isinstance(profiles, list):
        raise ValueError("plan must contain source roles and subject profiles")
    profile_by_subject = {
        int(row["subject"]): row
        for row in profiles
        if isinstance(row, dict) and row.get("subject") is not None
    }
    counts: list[int] | None = None
    for raw_subject in roles["source"]:
        subject = int(raw_subject)
        row = profile_by_subject.get(subject)
        if row is None or not isinstance(row.get("class_counts"), list) or not row["class_counts"]:
            raise ValueError(f"plan has no class counts for source subject {subject}")
        current = [int(item) for item in row["class_counts"]]
        if any(item < 0 for item in current):
            raise ValueError(f"negative class count for source subject {subject}")
        if counts is None:
            counts = [0] * len(current)
        if len(current) != len(counts):
            raise ValueError("source profiles use inconsistent class counts")
        counts = [left + right for left, right in zip(counts, current)]
    if not counts or sum(counts) <= 0:
        raise ValueError("source profiles contain no labels")
    return max(counts) / sum(counts)


def _cluster_grid(rows: Sequence[dict[str, Any]], value_key: str) -> tuple[list[int], list[int], dict[tuple[int, int], float]]:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        value = _finite(row.get(value_key))
        if value is None:
            continue
        grouped[(int(row["seed"]), int(row["subject"]))].append(value)
    grid = {key: _mean(values) for key, values in grouped.items()}
    seeds = sorted({key[0] for key in grid})
    subjects = sorted({key[1] for key in grid})
    return seeds, subjects, grid


def two_way_seed_subject_cluster_bootstrap_ci(
    rows: Sequence[dict[str, Any]],
    *,
    value_key: str = "fresh_gap",
    repeats: int = 2000,
    bootstrap_seed: int = 20260904,
    alpha: float = 0.05,
) -> list[float] | None:
    """Return a percentile CI after independently resampling both clusters.

    A replicate samples the observed seed IDs with replacement and independently
    samples target-subject IDs with replacement.  Every sampled seed-subject
    crossing contributes its cell mean; repeated order observations within a
    crossing are averaged first.  This preserves both dependence dimensions
    without treating stages or orders as independent replicates.
    """

    seeds, subjects, grid = _cluster_grid(rows, value_key)
    if len(seeds) < 2 or len(subjects) < 2 or repeats < 1 or not 0.0 < alpha < 1.0:
        return None
    rng = random.Random(int(bootstrap_seed))
    estimates: list[float] = []
    for _ in range(int(repeats)):
        sampled_seeds = [rng.choice(seeds) for _ in seeds]
        sampled_subjects = [rng.choice(subjects) for _ in subjects]
        values = [grid[(seed, subject)] for seed in sampled_seeds for subject in sampled_subjects if (seed, subject) in grid]
        if values:
            estimates.append(_mean(values))
    if not estimates:
        return None
    return [
        _percentile(estimates, alpha / 2.0),
        _percentile(estimates, 1.0 - alpha / 2.0),
    ]


def _seed_sign_flip_p_value(rows: Sequence[dict[str, Any]], *, value_key: str, alternative: str, random_seed: int) -> float | None:
    by_seed: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        value = _finite(row.get(value_key))
        if value is not None:
            by_seed[int(row["seed"])].append(value)
    seed_values = [_mean(by_seed[seed]) for seed in sorted(by_seed)]
    if len(seed_values) < 2:
        return None
    observed = _mean(seed_values)
    if len(seed_values) <= 16:
        signs: Iterable[tuple[int, ...]] = itertools.product((-1, 1), repeat=len(seed_values))
    else:
        rng = random.Random(int(random_seed))
        signs = ([rng.choice((-1, 1)) for _ in seed_values] for _ in range(65_536))
    extreme = 0
    total = 0
    tolerance = 1e-15
    for sign_row in signs:
        statistic = _mean(sign * value for sign, value in zip(sign_row, seed_values))
        total += 1
        if alternative == "greater":
            extreme += statistic >= observed - tolerance
        elif alternative == "two-sided":
            extreme += abs(statistic) >= abs(observed) - tolerance
        else:
            raise ValueError("alternative must be greater or two-sided")
    return extreme / total


def holm_adjust(p_values: Sequence[float | None], *, alpha: float = 0.05) -> list[dict[str, Any] | None]:
    """Apply Holm's step-down correction while preserving input order."""

    present = [(index, float(value)) for index, value in enumerate(p_values) if value is not None and math.isfinite(float(value))]
    ordered = sorted(present, key=lambda item: (item[1], item[0]))
    adjusted: dict[int, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[index] = running
    return [
        None if index not in adjusted else {"adjusted_p_value": adjusted[index], "reject": adjusted[index] <= alpha}
        for index in range(len(p_values))
    ]


def _slope(rows: Sequence[dict[str, Any]]) -> float | None:
    points = [(_finite(row.get("stage")), _finite(row.get("fresh_gap"))) for row in rows]
    valid = [(x, y) for x, y in points if x is not None and y is not None]
    if len(valid) < 2:
        return None
    mean_x = _mean(item[0] for item in valid)
    mean_y = _mean(item[1] for item in valid)
    denominator = sum((x - mean_x) ** 2 for x, _ in valid)
    if denominator == 0.0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in valid) / denominator


def _summary_statistics(
    rows: Sequence[dict[str, Any]],
    *,
    bootstrap_repeats: int,
    bootstrap_seed: int,
    alpha: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["architecture"]), int(row["budget"]))].append(row)
    summaries: list[dict[str, Any]] = []
    for index, ((architecture, budget), selected) in enumerate(sorted(grouped.items())):
        values = [float(row["fresh_gap"]) for row in selected]
        raw_p = _seed_sign_flip_p_value(selected, value_key="fresh_gap", alternative="greater", random_seed=bootstrap_seed + index)
        summaries.append({
            "architecture": architecture,
            "budget": budget,
            "valid_cell_count": len(values),
            "order_count": len({str(row["order"]) for row in selected}),
            "seed_count": len({int(row["seed"]) for row in selected}),
            "target_subject_count": len({int(row["subject"]) for row in selected}),
            "mean_fresh_gap": _mean(values),
            "median_fresh_gap": statistics.median(values),
            "positive_count": sum(value > 0.0 for value in values),
            "positive_fraction": sum(value > 0.0 for value in values) / len(values),
            "two_way_seed_subject_cluster_bootstrap_ci95": two_way_seed_subject_cluster_bootstrap_ci(
                selected,
                repeats=bootstrap_repeats,
                bootstrap_seed=bootstrap_seed + 1009 * index,
                alpha=alpha,
            ),
            "primary_test": budget > 0,
            "raw_p_value": raw_p if budget > 0 else None,
            "p_value_method": "one-sided exact seed-cluster sign-flip; H1: mean fresh_gap > 0" if budget > 0 else None,
        })
    primary_indexes = [index for index, row in enumerate(summaries) if row["primary_test"]]
    corrected = holm_adjust([summaries[index]["raw_p_value"] for index in primary_indexes], alpha=alpha)
    for index, correction in zip(primary_indexes, corrected):
        summaries[index]["holm_family"] = "all positive-budget architecture-by-budget primary tests"
        summaries[index]["holm_adjusted_p_value"] = None if correction is None else correction["adjusted_p_value"]
        summaries[index]["holm_reject"] = False if correction is None else correction["reject"]
    for row in summaries:
        if not row["primary_test"]:
            row.update({"holm_family": None, "holm_adjusted_p_value": None, "holm_reject": False})
    return summaries


def _order_effects(
    rows: Sequence[dict[str, Any]],
    expected_orders: Sequence[str],
    *,
    bootstrap_repeats: int,
    bootstrap_seed: int,
    alpha: float,
) -> list[dict[str, Any]]:
    if len(expected_orders) < 2:
        return []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["architecture"]), int(row["budget"]))].append(row)
    results: list[dict[str, Any]] = []
    for group_index, ((architecture, budget), selected) in enumerate(sorted(grouped.items())):
        for contrast_index, (left, right) in enumerate(itertools.combinations(expected_orders, 2)):
            by_cell: dict[tuple[int, int, str], list[float]] = defaultdict(list)
            for row in selected:
                by_cell[(int(row["seed"]), int(row["subject"]), str(row["order"]))].append(float(row["fresh_gap"]))
            differences = []
            for seed, subject in sorted({(key[0], key[1]) for key in by_cell}):
                left_values = by_cell.get((seed, subject, left))
                right_values = by_cell.get((seed, subject, right))
                if left_values and right_values:
                    differences.append({"seed": seed, "subject": subject, "difference": _mean(left_values) - _mean(right_values)})
            left_rows = [row for row in selected if row["order"] == left]
            right_rows = [row for row in selected if row["order"] == right]
            left_slope = _slope(left_rows)
            right_slope = _slope(right_rows)
            values = [float(row["difference"]) for row in differences]
            raw_p = _seed_sign_flip_p_value(
                differences,
                value_key="difference",
                alternative="two-sided",
                random_seed=bootstrap_seed + group_index * 1009 + contrast_index,
            )
            results.append({
                "architecture": architecture,
                "budget": budget,
                "contrast": f"{left} - {right}",
                "left_order": left,
                "right_order": right,
                "paired_seed_subject_count": len(differences),
                "seed_count": len({row["seed"] for row in differences}),
                "target_subject_count": len({row["subject"] for row in differences}),
                "mean_paired_difference": None if not values else _mean(values),
                "median_paired_difference": None if not values else statistics.median(values),
                "two_way_seed_subject_cluster_bootstrap_ci95": two_way_seed_subject_cluster_bootstrap_ci(
                    differences,
                    value_key="difference",
                    repeats=bootstrap_repeats,
                    bootstrap_seed=bootstrap_seed + 100_003 + group_index * 1009 + contrast_index,
                    alpha=alpha,
                ),
                "raw_p_value": raw_p,
                "p_value_method": "two-sided exact seed-cluster sign-flip on subject-paired order differences",
                "left_stage_slope": left_slope,
                "right_stage_slope": right_slope,
                "order_by_stage_slope_interaction": None if left_slope is None or right_slope is None else left_slope - right_slope,
            })
    corrected = holm_adjust([row["raw_p_value"] for row in results], alpha=alpha)
    for row, correction in zip(results, corrected):
        row["holm_family"] = "exploratory paired order contrasts"
        row["holm_adjusted_p_value"] = None if correction is None else correction["adjusted_p_value"]
        row["holm_reject"] = False if correction is None else correction["reject"]
    return results


def _expected_design(plan: dict[str, Any], config: dict[str, Any], phase_name: str) -> dict[str, Any]:
    phases = config.get("phases")
    if not isinstance(phases, dict) or not isinstance(phases.get(phase_name), dict):
        raise ValueError(f"config has no phase {phase_name!r}")
    phase = phases[phase_name]
    plan_orders = plan.get("orders")
    if not isinstance(plan_orders, dict):
        raise ValueError("plan has no orders")
    order_names = [str(item) for item in phase.get("orders", [])]
    if not order_names:
        raise ValueError("phase has no expected orders")
    target_limit = phase.get("target_limit")
    orders: dict[str, list[int]] = {}
    for name in order_names:
        if not isinstance(plan_orders.get(name), list):
            raise ValueError(f"plan has no order {name!r}")
        subjects = [int(item) for item in plan_orders[name]]
        orders[name] = subjects if target_limit is None else subjects[: int(target_limit)]
    architectures = [str(item) for item in phase.get("architectures", [])]
    seeds = [int(item) for item in phase.get("seeds", [])]
    budgets = [int(item) for item in phase.get("budgets", [])]
    if not architectures or not seeds or not budgets:
        raise ValueError("phase architectures, seeds and budgets must be non-empty")
    if budgets != sorted(set(budgets)) or budgets[0] != 0:
        raise ValueError("phase budgets must be sorted, unique and start at zero")
    return {"phase": phase, "order_names": order_names, "orders": orders, "architectures": architectures, "seeds": seeds, "budgets": budgets}


def _metadata_mismatches(metadata: dict[str, Any], plan: dict[str, Any], design: dict[str, Any], order: str) -> list[str]:
    phase = design["phase"]
    roles = plan.get("roles") if isinstance(plan.get("roles"), dict) else {}
    expected = {
        "source_subjects": [int(item) for item in roles.get("source", [])],
        "target_subject_order": design["orders"][order],
        "retention_subjects": [int(item) for item in roles.get("retention", [])],
        "budgets": design["budgets"],
        "source_epochs": int(phase["source_epochs"]),
        "batch_size": int(phase["batch_size"]),
        "source_lr": float(phase["source_lr"]),
        "adapt_lr": float(phase["adapt_lr"]),
        "source_eval_fraction": float(phase["source_eval_fraction"]),
        "retention_max_samples": int(phase["retention_max_samples"]),
    }
    mismatches = []
    for key, value in expected.items():
        actual = metadata.get(key)
        if isinstance(value, float):
            actual_number = _finite(actual)
            matches = actual_number is not None and math.isclose(actual_number, value, rel_tol=1e-12, abs_tol=1e-12)
        else:
            matches = actual == value
        if not matches:
            mismatches.append(key)
    if metadata.get("schema") != "edgeforge.eeg-continuous-architecture-lop.v1":
        mismatches.append("schema")
    return mismatches


def analyze(
    summary_inputs: Sequence[tuple[str, str | Path]],
    plan: dict[str, Any],
    config: dict[str, Any],
    *,
    phase_name: str,
    bootstrap_repeats: int | None = None,
    bootstrap_seed: int = 20260904,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    design = _expected_design(plan, config, phase_name)
    acceptance = config.get("acceptance")
    if not isinstance(acceptance, dict):
        acceptance = {}
    source_margin = float(acceptance.get("source_accuracy_above_majority", 0.1))
    fresh_gain_minimum = float(acceptance.get("fresh_learning_accuracy_gain", 0.05))
    minimum_seeds = int(acceptance.get("minimum_independent_seeds", 10))
    minimum_stages = int(acceptance.get("minimum_target_stages", 50))
    alpha = float(acceptance.get("alpha", 0.05))
    repeats = int(bootstrap_repeats if bootstrap_repeats is not None else acceptance.get("bootstrap_repeats", 10_000))
    if repeats < 1:
        raise ValueError("bootstrap_repeats must be positive")
    majority = source_majority_accuracy(plan)
    issues: list[dict[str, Any]] = []

    if manifest is not None:
        for record in manifest.get("commands", []):
            if isinstance(record, dict) and record.get("status") != "succeeded":
                issues.append({"code": "manifest_command_not_succeeded", "order": record.get("order"), "architecture": record.get("architecture"), "status": record.get("status")})

    parsed_runs: dict[tuple[str, str, int], dict[str, Any]] = {}
    run_audits: dict[tuple[str, str, int], dict[str, Any]] = {}
    input_records: list[dict[str, Any]] = []
    for raw_order, raw_path in summary_inputs:
        order = str(raw_order)
        path = Path(raw_path).resolve()
        input_record = {"order": order, "path": str(path), "loaded": False}
        input_records.append(input_record)
        if order not in design["orders"]:
            issues.append({"code": "unexpected_summary_order", "order": order, "path": str(path)})
            continue
        try:
            summary = _load_object(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            issues.append({"code": "summary_unreadable", "order": order, "path": str(path), "detail": str(error)})
            continue
        input_record["loaded"] = True
        metadata = summary.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        mismatches = _metadata_mismatches(metadata, plan, design, order)
        input_record["metadata_mismatches"] = mismatches
        if mismatches:
            issues.append({"code": "summary_design_mismatch", "order": order, "path": str(path), "fields": mismatches})
        runs = summary.get("runs")
        if not isinstance(runs, list):
            issues.append({"code": "summary_runs_not_list", "order": order, "path": str(path)})
            continue
        if summary.get("completed_runs") != len(runs) or summary.get("planned_runs") != len(runs):
            issues.append({"code": "summary_incomplete", "order": order, "path": str(path), "planned_runs": summary.get("planned_runs"), "completed_runs": summary.get("completed_runs"), "stored_runs": len(runs)})
        for run in runs:
            if not isinstance(run, dict):
                issues.append({"code": "invalid_run_record", "order": order, "path": str(path)})
                continue
            try:
                architecture = str(run["architecture"])
                seed = int(run["seed"])
            except (KeyError, TypeError, ValueError):
                issues.append({"code": "invalid_run_identity", "order": order, "path": str(path)})
                continue
            key = (architecture, order, seed)
            if key in parsed_runs:
                issues.append({"code": "duplicate_run", "architecture": architecture, "order": order, "seed": seed})
                run_audits[key]["duplicate"] = True
                continue
            parsed_runs[key] = run
            run_audits[key] = {"architecture": architecture, "order": order, "seed": seed, "source_accuracy": None, "source_accuracy_delta_above_majority": None, "source_learning_adequate": False, "duplicate": False, "summary_design_valid": not mismatches}

    expected_runs = {
        (architecture, order, seed)
        for architecture in design["architectures"]
        for order in design["order_names"]
        for seed in design["seeds"]
    }
    for key in sorted(set(parsed_runs) - expected_runs):
        issues.append({"code": "unexpected_run", "architecture": key[0], "order": key[1], "seed": key[2]})
    for key in sorted(expected_runs - set(parsed_runs)):
        architecture, order, seed = key
        issues.append({"code": "missing_run", "architecture": architecture, "order": order, "seed": seed})
        run_audits[key] = {"architecture": architecture, "order": order, "seed": seed, "missing": True, "source_learning_adequate": False, "summary_design_valid": False}

    cells: dict[tuple[str, str, int, int, int, int], dict[str, Any]] = {}
    stage_audits: list[dict[str, Any]] = []
    for run_key in sorted(expected_runs & set(parsed_runs)):
        architecture, order, seed = run_key
        run = parsed_runs[run_key]
        run_audit = run_audits[run_key]
        source = run.get("source")
        source_accuracy = _finite(source.get("accuracy")) if isinstance(source, dict) else None
        source_delta = None if source_accuracy is None else source_accuracy - majority
        source_adequate = source_delta is not None and source_delta >= source_margin
        run_audit.update({"source_accuracy": source_accuracy, "source_majority_accuracy": majority, "source_accuracy_delta_above_majority": source_delta, "required_source_accuracy_delta": source_margin, "source_learning_adequate": source_adequate})
        if run.get("status") != "complete":
            issues.append({"code": "run_not_complete", "architecture": architecture, "order": order, "seed": seed, "status": run.get("status")})
        if source_accuracy is None:
            issues.append({"code": "missing_source_accuracy", "architecture": architecture, "order": order, "seed": seed})
        elif not source_adequate:
            issues.append({"code": "source_learning_insufficient", "architecture": architecture, "order": order, "seed": seed, "accuracy": source_accuracy, "majority": majority, "required_delta": source_margin})
        stages = run.get("stages")
        if not isinstance(stages, list):
            stages = []
            issues.append({"code": "run_stages_not_list", "architecture": architecture, "order": order, "seed": seed})
        stage_by_index: dict[int, dict[str, Any]] = {}
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            try:
                stage_index = int(stage["stage"])
            except (KeyError, TypeError, ValueError):
                issues.append({"code": "invalid_stage_index", "architecture": architecture, "order": order, "seed": seed})
                continue
            if stage_index in stage_by_index:
                issues.append({"code": "duplicate_stage", "architecture": architecture, "order": order, "seed": seed, "stage": stage_index})
                continue
            stage_by_index[stage_index] = stage
        expected_subjects = design["orders"][order]
        for stage_index in sorted(set(stage_by_index) - set(range(len(expected_subjects)))):
            issues.append({"code": "unexpected_stage", "architecture": architecture, "order": order, "seed": seed, "stage": stage_index})
        for stage_index, subject in enumerate(expected_subjects):
            stage = stage_by_index.get(stage_index)
            audit = {"architecture": architecture, "order": order, "seed": seed, "stage": stage_index, "subject": subject, "fresh_initial_accuracy": None, "fresh_final_accuracy": None, "fresh_learning_accuracy_gain": None, "required_fresh_learning_accuracy_gain": fresh_gain_minimum, "fresh_learning_adequate": False, "reasons": []}
            stage_audits.append(audit)
            if stage is None:
                audit["reasons"].append("missing_stage")
                issues.append({"code": "missing_stage", "architecture": architecture, "order": order, "seed": seed, "stage": stage_index, "subject": subject})
                continue
            if stage.get("subject") != subject:
                audit["reasons"].append("subject_mismatch")
                issues.append({"code": "stage_subject_mismatch", "architecture": architecture, "order": order, "seed": seed, "stage": stage_index, "expected_subject": subject, "actual_subject": stage.get("subject")})
            fresh_rows, fresh_reasons = _index_rows(stage.get("fresh", {}).get("curve") if isinstance(stage.get("fresh"), dict) else None, field="fresh_curve", context="stage")
            warm_rows, warm_reasons = _index_rows(stage.get("warm", {}).get("curve") if isinstance(stage.get("warm"), dict) else None, field="warm_curve", context="stage")
            gap_rows, gap_reasons = _index_rows(stage.get("gaps"), field="gaps", context="stage")
            audit["reasons"].extend(fresh_reasons + warm_reasons + gap_reasons)
            expected_budget_set = set(design["budgets"])
            for family, indexed in (("fresh", fresh_rows), ("warm", warm_rows), ("gap", gap_rows)):
                unexpected_budgets = sorted(set(indexed) - expected_budget_set)
                if unexpected_budgets:
                    audit["reasons"].append(f"unexpected_{family}_budgets")
                    issues.append({"code": "unexpected_budget", "family": family, "architecture": architecture, "order": order, "seed": seed, "stage": stage_index, "budgets": unexpected_budgets})
            first_fresh = fresh_rows.get(design["budgets"][0])
            final_fresh = fresh_rows.get(design["budgets"][-1])
            initial_accuracy = _finite(first_fresh.get("accuracy")) if first_fresh else None
            final_accuracy = _finite(final_fresh.get("accuracy")) if final_fresh else None
            gain = None if initial_accuracy is None or final_accuracy is None else final_accuracy - initial_accuracy
            fresh_adequate = gain is not None and gain >= fresh_gain_minimum
            audit.update({"fresh_initial_accuracy": initial_accuracy, "fresh_final_accuracy": final_accuracy, "fresh_learning_accuracy_gain": gain, "fresh_learning_adequate": fresh_adequate})
            if gain is None:
                audit["reasons"].append("missing_fresh_learning_gain")
                issues.append({"code": "missing_fresh_learning_gain", "architecture": architecture, "order": order, "seed": seed, "stage": stage_index, "subject": subject})
            elif not fresh_adequate:
                audit["reasons"].append("fresh_learning_insufficient")
                issues.append({"code": "fresh_learning_insufficient", "architecture": architecture, "order": order, "seed": seed, "stage": stage_index, "subject": subject, "gain": gain, "required_gain": fresh_gain_minimum})
            for budget in design["budgets"]:
                cell_key = (architecture, order, seed, stage_index, subject, budget)
                reasons = list(audit["reasons"])
                if not run_audit["summary_design_valid"]:
                    reasons.append("summary_design_mismatch")
                if run.get("status") != "complete":
                    reasons.append("run_not_complete")
                if run_audit.get("duplicate"):
                    reasons.append("duplicate_run")
                if not source_adequate:
                    reasons.append("source_learning_insufficient")
                fresh_row = fresh_rows.get(budget)
                warm_row = warm_rows.get(budget)
                gap_row = gap_rows.get(budget)
                fresh_accuracy = _finite(fresh_row.get("accuracy")) if fresh_row else None
                warm_accuracy = _finite(warm_row.get("accuracy")) if warm_row else None
                fresh_gap = _finite(gap_row.get("fresh_gap")) if gap_row else None
                if fresh_row is None:
                    reasons.append("missing_fresh_budget")
                if warm_row is None:
                    reasons.append("missing_warm_budget")
                if gap_row is None or fresh_gap is None:
                    reasons.append("missing_fresh_gap")
                if fresh_accuracy is not None and warm_accuracy is not None and fresh_gap is not None and not math.isclose(fresh_gap, fresh_accuracy - warm_accuracy, rel_tol=1e-9, abs_tol=1e-9):
                    reasons.append("fresh_gap_inconsistent")
                cells[cell_key] = {
                    "architecture": architecture,
                    "order": order,
                    "seed": seed,
                    "stage": stage_index,
                    "subject": subject,
                    "budget": budget,
                    "fresh_gap": fresh_gap,
                    "fresh_accuracy": fresh_accuracy,
                    "warm_accuracy": warm_accuracy,
                    "source_accuracy": source_accuracy,
                    "source_majority_accuracy": majority,
                    "source_accuracy_delta_above_majority": source_delta,
                    "fresh_learning_accuracy_gain": gain,
                    "valid": not reasons,
                    "invalid_reasons": sorted(set(reasons)),
                }

    expected_cells = {
        (architecture, order, seed, stage, subject, budget)
        for architecture in design["architectures"]
        for order in design["order_names"]
        for seed in design["seeds"]
        for stage, subject in enumerate(design["orders"][order])
        for budget in design["budgets"]
    }
    for key in sorted(expected_cells - set(cells)):
        architecture, order, seed, stage, subject, budget = key
        cells[key] = {"architecture": architecture, "order": order, "seed": seed, "stage": stage, "subject": subject, "budget": budget, "fresh_gap": None, "fresh_accuracy": None, "warm_accuracy": None, "source_accuracy": None, "source_majority_accuracy": majority, "source_accuracy_delta_above_majority": None, "fresh_learning_accuracy_gain": None, "valid": False, "invalid_reasons": ["missing_cell"]}

    valid_cells = [row for row in cells.values() if row["valid"]]
    cell_rows = [cells[key] for key in sorted(cells)]
    statistics_rows = _summary_statistics(valid_cells, bootstrap_repeats=repeats, bootstrap_seed=bootstrap_seed, alpha=alpha)
    order_rows = _order_effects(valid_cells, design["order_names"], bootstrap_repeats=repeats, bootstrap_seed=bootstrap_seed, alpha=alpha)
    complete = set(parsed_runs) >= expected_runs and len(cells) == len(expected_cells) and not any(
        item["code"] in {"missing_run", "missing_stage", "unexpected_stage", "unexpected_budget", "summary_incomplete", "summary_unreadable", "summary_runs_not_list", "duplicate_run", "manifest_command_not_succeeded"}
        for item in issues
    )
    design_valid = not any(item["code"] in {"unexpected_summary_order", "unexpected_run", "summary_design_mismatch", "stage_subject_mismatch"} for item in issues)
    learning_adequate = bool(expected_runs) and all(run_audits[key].get("source_learning_adequate", False) for key in expected_runs) and len(stage_audits) == sum(len(design["orders"][order]) for order in design["order_names"]) * len(design["architectures"]) * len(design["seeds"]) and all(row["fresh_learning_adequate"] for row in stage_audits)
    sample_size_adequate = len(set(design["seeds"])) >= minimum_seeds and all(len(design["orders"][order]) >= minimum_stages for order in design["order_names"])
    every_cell_valid = len(valid_cells) == len(expected_cells)
    candidate_ready = complete and design_valid and learning_adequate and sample_size_adequate and every_cell_valid
    status = "candidate-evidence-ready" if candidate_ready else "blocked"
    return {
        "schema": SCHEMA,
        "status": status,
        "phase": phase_name,
        "experiment_version": config.get("experiment_version"),
        "plan_digest": plan.get("plan_digest"),
        "inputs": input_records,
        "expected_design": {"orders": design["orders"], "architectures": design["architectures"], "seeds": design["seeds"], "budgets": design["budgets"], "expected_runs": len(expected_runs), "expected_cells": len(expected_cells)},
        "acceptance": {"source_majority_accuracy": majority, "source_accuracy_above_majority": source_margin, "fresh_learning_accuracy_gain": fresh_gain_minimum, "minimum_independent_seeds": minimum_seeds, "minimum_target_stages": minimum_stages, "bootstrap_repeats": repeats, "alpha": alpha},
        "audit": {"completeness_passed": complete, "design_consistency_passed": design_valid, "learning_adequacy_passed": learning_adequate, "sample_size_passed": sample_size_adequate, "all_expected_cells_valid": every_cell_valid, "expected_run_count": len(expected_runs), "observed_expected_run_count": len(expected_runs & set(parsed_runs)), "expected_cell_count": len(expected_cells), "valid_cell_count": len(valid_cells), "invalid_cell_count": len(expected_cells) - len(valid_cells), "issue_count": len(issues), "issues": issues},
        "run_audit": [run_audits[key] for key in sorted(run_audits)],
        "stage_learning_audit": stage_audits,
        "cells": cell_rows,
        "valid_cells": valid_cells,
        "architecture_budget_statistics": statistics_rows,
        "order_effects": order_rows,
        "inference": {"bootstrap_method": BOOTSTRAP_METHOD, "bootstrap_description": "Seed IDs and target-subject IDs are sampled independently with replacement; repeated order observations within each sampled crossing are averaged.", "primary_p_value_method": "one-sided exact seed-cluster sign-flip", "multiple_comparison": "Holm step-down across positive-budget architecture-by-budget primary tests", "order_contrasts": "Subject-paired order differences plus order-by-stage OLS slope interaction; order-contrast Holm family is exploratory and separate."},
        "candidate_evidence_ready": candidate_ready,
        "conclusion_scope": "candidate only; requires scientific review and independent replication",
        "scientific_conclusion_allowed": False,
    }


def render_markdown(result: dict[str, Any]) -> str:
    audit = result["audit"]
    lines = [
        "# Full EEG LoP experiment analysis",
        "",
        f"- Status: `{result['status']}`",
        f"- Phase: `{result['phase']}`",
        f"- Candidate evidence ready: `{result['candidate_evidence_ready']}`",
        f"- Scientific conclusion allowed: `{result['scientific_conclusion_allowed']}`",
        f"- Valid cells: {audit['valid_cell_count']}/{audit['expected_cell_count']}",
        f"- Bootstrap: `{result['inference']['bootstrap_method']}`",
        "",
        "Passing this report means candidate evidence is complete enough for review. It is not an automatic scientific conclusion.",
        "",
        "## Audit gates",
        "",
        "| Gate | Passed |",
        "| --- | --- |",
        f"| Completeness | `{audit['completeness_passed']}` |",
        f"| Design consistency | `{audit['design_consistency_passed']}` |",
        f"| Source and fresh learning adequacy | `{audit['learning_adequacy_passed']}` |",
        f"| Minimum seeds and target stages | `{audit['sample_size_passed']}` |",
        f"| Every expected cell valid | `{audit['all_expected_cells_valid']}` |",
        "",
        "## Architecture by budget",
        "",
        "Only valid cells appear in this table.",
        "",
        "| Architecture | Budget | n | Seeds | Subjects | Mean gap | Median gap | Positive fraction | Two-way cluster 95% CI | Holm p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in result["architecture_budget_statistics"]:
        ci = row["two_way_seed_subject_cluster_bootstrap_ci95"]
        ci_text = "n/a" if ci is None else f"[{ci[0]:.6g}, {ci[1]:.6g}]"
        holm = row["holm_adjusted_p_value"]
        lines.append(f"| `{row['architecture']}` | {row['budget']} | {row['valid_cell_count']} | {row['seed_count']} | {row['target_subject_count']} | {row['mean_fresh_gap']:.6g} | {row['median_fresh_gap']:.6g} | {row['positive_fraction']:.3f} | {ci_text} | {'n/a' if holm is None else f'{holm:.6g}'} |")
    lines.extend(["", "## Order effects", "", "Differences match the same seed and target subject across orders. The slope interaction is `(left order stage slope) - (right order stage slope)`.", "", "| Architecture | Budget | Contrast | Pairs | Mean difference | Two-way cluster 95% CI | Stage-slope interaction | Holm p |", "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in result["order_effects"]:
        ci = row["two_way_seed_subject_cluster_bootstrap_ci95"]
        ci_text = "n/a" if ci is None else f"[{ci[0]:.6g}, {ci[1]:.6g}]"
        mean = row["mean_paired_difference"]
        interaction = row["order_by_stage_slope_interaction"]
        holm = row["holm_adjusted_p_value"]
        lines.append(f"| `{row['architecture']}` | {row['budget']} | `{row['contrast']}` | {row['paired_seed_subject_count']} | {'n/a' if mean is None else f'{mean:.6g}'} | {ci_text} | {'n/a' if interaction is None else f'{interaction:.6g}'} | {'n/a' if holm is None else f'{holm:.6g}'} |")
    lines.extend(["", "## Audit issues", ""])
    if not audit["issues"]:
        lines.append("- None.")
    else:
        counts: dict[str, int] = defaultdict(int)
        for issue in audit["issues"]:
            counts[str(issue["code"])] += 1
        for code, count in sorted(counts.items()):
            lines.append(f"- `{code}`: {count}")
    lines.append("")
    return "\n".join(lines)


def _resolve_manifest_path(raw: Any, manifest_path: Path) -> Path:
    path = Path(str(raw)).expanduser()
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def summaries_from_manifest(manifest: dict[str, Any], manifest_path: Path) -> list[tuple[str, Path]]:
    records: list[tuple[str, Path]] = []
    commands = manifest.get("commands")
    if not isinstance(commands, list):
        raise ValueError("run manifest has no commands list")
    for record in commands:
        if not isinstance(record, dict) or record.get("order") is None or record.get("output") is None:
            continue
        output = _resolve_manifest_path(record["output"], manifest_path)
        records.append((str(record["order"]), output / "summary.json"))
    return records


def _parse_summary_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--summary must use ORDER=PATH")
    order, raw_path = value.split("=", 1)
    if not order or not raw_path:
        raise argparse.ArgumentTypeError("--summary must use non-empty ORDER=PATH")
    return order, Path(raw_path).expanduser()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="append", type=_parse_summary_argument, default=[], metavar="ORDER=PATH")
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--phase", choices=("calibration", "confirmatory", "data-composition"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int)
    parser.add_argument("--bootstrap-seed", type=int, default=20260904)
    args = parser.parse_args()
    if bool(args.summary) == bool(args.run_manifest):
        parser.error("provide either repeated --summary ORDER=PATH or one --run-manifest")

    manifest = None
    if args.run_manifest:
        manifest_path = args.run_manifest.resolve()
        manifest = _load_object(manifest_path)
        summaries = summaries_from_manifest(manifest, manifest_path)
        plan_path = args.plan.resolve() if args.plan else _resolve_manifest_path(manifest.get("plan"), manifest_path)
        config_path = args.config.resolve() if args.config else _resolve_manifest_path(manifest.get("config"), manifest_path)
        phase_name = args.phase or str(manifest.get("phase") or "")
    else:
        if not args.plan or not args.config or not args.phase:
            parser.error("explicit --summary inputs also require --plan, --config and --phase")
        summaries = args.summary
        plan_path = args.plan.resolve()
        config_path = args.config.resolve()
        phase_name = args.phase
    if not phase_name:
        parser.error("phase is missing")
    result = analyze(
        summaries,
        _load_object(plan_path),
        _load_object(config_path),
        phase_name=phase_name,
        bootstrap_repeats=args.bootstrap_repeats,
        bootstrap_seed=args.bootstrap_seed,
        manifest=manifest,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "analysis.json"
    markdown_path = output / "analysis.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "json": str(json_path), "markdown": str(markdown_path), "valid_cells": result["audit"]["valid_cell_count"], "expected_cells": result["audit"]["expected_cell_count"], "candidate_evidence_ready": result["candidate_evidence_ready"], "scientific_conclusion_allowed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
