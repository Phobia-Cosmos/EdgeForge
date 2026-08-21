"""Deterministic, descriptive RA-EEG LoP analysis helpers.

The analysis deliberately reports evidence quality and never turns a small
correlation into a scientific or causal conclusion.  It is dependency-free so
the control plane can validate and persist an analysis contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from typing import Any, Iterable


ANALYSIS_SCHEMA_VERSION = 1
DEFAULT_PREDICTOR = "task.spectra.transformer_1.effective_rank"
DEFAULT_OUTCOME = "plasticity.acc_gain"
SCIENTIFIC_MINIMUM_SEEDS = 3


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _mean(values: Iterable[float]) -> float:
    items = sorted(values)
    return sum(items) / len(items)


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mean_x, mean_y = _mean(x), _mean(y)
    centered_x = [value - mean_x for value in x]
    centered_y = [value - mean_y for value in y]
    denominator = math.sqrt(sum(value * value for value in centered_x) * sum(value * value for value in centered_y))
    if denominator == 0:
        return None
    correlation = sum(left * right for left, right in zip(centered_x, centered_y)) / denominator
    return min(1.0, max(-1.0, correlation))


def _rank(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            ranks[ordered[position][0]] = rank
        index = end
    return ranks


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    return _pearson(_rank(x), _rank(y))


def _percentile(values: list[float], probability: float) -> float:
    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _bootstrap_interval(pairs: list[dict[str, Any]], statistic: str, *, seed: int, repeats: int) -> list[float] | None:
    if len(pairs) < 3:
        return None
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        cluster = f"seed:{pair['seed']}" if isinstance(pair.get("seed"), int) else f"experiment:{pair['experiment_id']}"
        clusters[cluster].append(pair)
    if len(clusters) < 2:
        return None
    rng = random.Random(seed)
    values: list[float] = []
    cluster_names = sorted(clusters)
    for _ in range(repeats):
        sample_pairs = [pair for _ in cluster_names for pair in clusters[rng.choice(cluster_names)]]
        sample_x = [pair["predictor"] for pair in sample_pairs]
        sample_y = [pair["outcome"] for pair in sample_pairs]
        result = _pearson(sample_x, sample_y) if statistic == "pearson" else _spearman(sample_x, sample_y)
        if result is not None and math.isfinite(result):
            values.append(result)
    if len(values) < 10:
        return None
    values.sort()
    return [round(_percentile(values, 0.025), 8), round(_percentile(values, 0.975), 8)]


def _context_key(context: Any) -> str:
    return json.dumps(context if isinstance(context, dict) else {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _step_values(metrics: list[dict[str, Any]], name: str) -> dict[int, list[float]]:
    values: dict[int, list[float]] = defaultdict(list)
    for metric in metrics:
        if metric.get("name") != name or metric.get("step") is None:
            continue
        value = _finite(metric.get("value"))
        if value is not None:
            values[int(metric["step"])].append(value)
    return values


def _exact_values(metrics: list[dict[str, Any]], name: str) -> dict[tuple[int, str], float]:
    values: dict[tuple[int, str], list[float]] = defaultdict(list)
    for metric in metrics:
        if metric.get("name") != name or metric.get("step") is None:
            continue
        value = _finite(metric.get("value"))
        if value is not None:
            values[(int(metric["step"]), _context_key(metric.get("context")))].append(value)
    return {key: _mean(items) for key, items in values.items()}


def _ordered_stage_pairs(
    predictor_values: dict[int, float], outcome_values: dict[int, float], lag: int
) -> list[tuple[int, int]]:
    stages = sorted(set(predictor_values) | set(outcome_values))
    return [
        (stages[index - lag], stages[index])
        for index in range(lag, len(stages))
        if stages[index - lag] in predictor_values and stages[index] in outcome_values
    ]


def analyze_lop(
    experiments: list[dict[str, Any]],
    metrics_by_experiment: dict[str, list[dict[str, Any]]],
    *,
    predictor: str = DEFAULT_PREDICTOR,
    outcome: str = DEFAULT_OUTCOME,
    lag: int = 1,
    context_policy: str = "aggregate-step",
    bootstrap_repeats: int = 2000,
    bootstrap_seed: int = 20260821,
    minimum_pairs: int = 3,
    minimum_seeds: int = 3,
) -> dict[str, Any]:
    if not predictor or not outcome:
        raise ValueError("predictor and outcome metric names are required")
    if lag < 0 or lag > 128:
        raise ValueError("lag must be between 0 and 128")
    if context_policy not in {"aggregate-step", "exact"}:
        raise ValueError("context_policy must be aggregate-step or exact")
    bootstrap_repeats = min(10_000, max(100, int(bootstrap_repeats)))
    minimum_pairs = min(100_000, max(2, int(minimum_pairs)))
    minimum_seeds = min(10_000, max(SCIENTIFIC_MINIMUM_SEEDS, int(minimum_seeds)))

    identities = []
    for experiment in sorted(experiments, key=lambda item: str(item.get("experiment_id") or "")):
        spec = experiment.get("spec") or {}
        metadata = spec.get("metadata") or {}
        identities.append({
            "experiment_id": experiment.get("experiment_id"),
            "task_id": experiment.get("task_id"),
            "version": experiment.get("version"),
            "runtime_version": experiment.get("runtime_version"),
            "artifact_digest": experiment.get("artifact_digest"),
            "source_digest": experiment.get("source_digest"),
            "workload": experiment.get("workload") or spec.get("workload"),
            "protocol": experiment.get("protocol") or spec.get("protocol"),
            "comparison_group": metadata.get("comparison_group") or spec.get("comparison_group"),
            "method": experiment.get("method") or spec.get("method"),
            "seed": experiment.get("seed", spec.get("seed")),
        })
    scopes = {
        (item["workload"], item["protocol"], item["comparison_group"], item["method"])
        for item in identities
    }
    scope_consistent = len(scopes) <= 1 and all(
        item["comparison_group"] and item["method"] for item in identities
    )

    pairs: list[dict[str, Any]] = []
    for identity in identities:
        experiment_id = str(identity["experiment_id"])
        metrics = metrics_by_experiment.get(experiment_id, [])
        if context_policy == "exact":
            predictor_values = _exact_values(metrics, predictor)
            outcome_values = _exact_values(metrics, outcome)
            contexts = sorted({context for _, context in predictor_values} | {context for _, context in outcome_values})
            for context in contexts:
                by_predictor = {step: value for (step, item_context), value in predictor_values.items() if item_context == context}
                by_outcome = {step: value for (step, item_context), value in outcome_values.items() if item_context == context}
                for source_step, outcome_step in _ordered_stage_pairs(by_predictor, by_outcome, lag):
                    pairs.append({"experiment_id": experiment_id, "seed": identity["seed"], "predictor_step": source_step, "outcome_step": outcome_step, "context": json.loads(context), "predictor": by_predictor[source_step], "outcome": by_outcome[outcome_step]})
        else:
            predictor_values = {step: _mean(values) for step, values in _step_values(metrics, predictor).items()}
            outcome_values = {step: _mean(values) for step, values in _step_values(metrics, outcome).items()}
            for source_step, outcome_step in _ordered_stage_pairs(predictor_values, outcome_values, lag):
                pairs.append({"experiment_id": experiment_id, "seed": identity["seed"], "predictor_step": source_step, "outcome_step": outcome_step, "context": {}, "predictor": predictor_values[source_step], "outcome": outcome_values[outcome_step]})

    x = [item["predictor"] for item in pairs]
    y = [item["outcome"] for item in pairs]
    seeds = sorted({item["seed"] for item in pairs if isinstance(item["seed"], int)})
    identity_seeds = [item["seed"] for item in identities if isinstance(item["seed"], int)]
    seeds_unique = len(identity_seeds) == len(set(identity_seeds))
    contributing_experiments = sorted({item["experiment_id"] for item in pairs})
    missing_experiments = sorted({str(item["experiment_id"]) for item in identities} - set(contributing_experiments))
    transitions_by_experiment = {
        str(identity["experiment_id"]): sorted({
            (pair["predictor_step"], pair["outcome_step"])
            for pair in pairs
            if pair["experiment_id"] == str(identity["experiment_id"])
        })
        for identity in identities
    }
    transition_signatures = {
        tuple(transitions_by_experiment[experiment_id])
        for experiment_id in contributing_experiments
    }
    stages_consistent = len(transition_signatures) <= 1
    pair_grids_by_experiment = {
        str(identity["experiment_id"]): sorted({
            (
                pair["predictor_step"],
                pair["outcome_step"],
                _context_key(pair["context"]),
            )
            for pair in pairs
            if pair["experiment_id"] == str(identity["experiment_id"])
        })
        for identity in identities
    }
    pair_grid_signatures = {
        tuple(pair_grids_by_experiment[experiment_id])
        for experiment_id in contributing_experiments
    }
    contexts_consistent = len(pair_grid_signatures) <= 1
    pearson = _pearson(x, y)
    spearman = _spearman(x, y)
    variation_sufficient = pearson is not None and spearman is not None
    status = "ok"
    reasons: list[str] = []
    if not scope_consistent:
        reasons.append("experiments must share workload, protocol, comparison_group and method")
    if not stages_consistent:
        reasons.append("experiments must share the same ordered checkpoint stage transitions")
    if not contexts_consistent:
        reasons.append("experiments must share the same lagged stage and context pairing grid")
    if not seeds_unique:
        reasons.append("each experiment must provide a unique real seed")
    if len(pairs) < minimum_pairs:
        status = "insufficient-pairs"
        reasons.append(f"pairs={len(pairs)} is below minimum_pairs={minimum_pairs}")
    if len(seeds) < minimum_seeds:
        reasons.append(f"seeds={len(seeds)} is below minimum_seeds={minimum_seeds}")
    if missing_experiments:
        reasons.append(f"experiments without usable pairs: {missing_experiments}")
    if len(pairs) >= minimum_pairs and not variation_sufficient:
        reasons.append("predictor and outcome must both vary enough to define Pearson and Spearman correlation")
    if not scope_consistent:
        status = "blocked-incomparable-scope"
    elif not stages_consistent:
        status = "blocked-incomparable-stages"
    elif not contexts_consistent:
        status = "blocked-incomparable-contexts"
    elif not seeds_unique:
        status = "blocked-duplicate-seeds"
    elif missing_experiments:
        status = "blocked-incomplete-evidence"
    elif len(pairs) < minimum_pairs:
        status = "insufficient-pairs"
    elif len(seeds) < minimum_seeds:
        status = "insufficient-seeds"
    elif not variation_sufficient:
        status = "insufficient-variation"
    result = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis": "lop-lagged-correlation-v1",
        "status": status,
        "scientific_conclusion_allowed": False,
        "scope_consistent": scope_consistent,
        "stages_consistent": stages_consistent,
        "contexts_consistent": contexts_consistent,
        "seeds_unique": seeds_unique,
        "variation_sufficient": variation_sufficient,
        "reasons": reasons,
        "predictor": predictor,
        "outcome": outcome,
        "lag": lag,
        "context_policy": context_policy,
        "minimum_pairs": minimum_pairs,
        "minimum_seeds": minimum_seeds,
        "bootstrap": {
            "method": "seed-cluster-percentile",
            "repeats": bootstrap_repeats,
            "seed": bootstrap_seed,
        },
        "experiment_count": len(identities),
        "seed_count": len(seeds),
        "seeds": seeds,
        "contributing_experiments": contributing_experiments,
        "missing_experiments": missing_experiments,
        "pair_count": len(pairs),
        "transitions_by_experiment": transitions_by_experiment,
        "pair_grids_by_experiment": pair_grids_by_experiment,
        "experiments": identities,
        "pairs": pairs,
        "statistics": {
            "predictor_mean": _mean(x) if x else None,
            "outcome_mean": _mean(y) if y else None,
            "pearson": pearson,
            "spearman": spearman,
            "pearson_ci95": _bootstrap_interval(pairs, "pearson", seed=bootstrap_seed, repeats=bootstrap_repeats),
            "spearman_ci95": _bootstrap_interval(pairs, "spearman", seed=bootstrap_seed + 1, repeats=bootstrap_repeats),
        },
        "interpretation": "descriptive association only; not a causal LoP conclusion",
    }
    result["analysis_digest"] = hashlib.sha256(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return result
