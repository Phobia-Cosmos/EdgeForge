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
DEFAULT_LOP_OUTCOME = "task.plasticity.fresh_gap"
SCIENTIFIC_MINIMUM_SEEDS = 3
PAIR_CONTEXT_KEYS = ("dataset", "subject", "split", "method", "layer", "task_context", "comparison_group")

# Results produced by different RA-EEG generations used slightly different
# envelope prefixes.  Keep the v1 defaults stable for API compatibility, but
# resolve these aliases at analysis time so an imported BrainUICL probe with
# ``task.spectra.transformer.effective_rank`` and ``task.plasticity.acc_gain``
# is not incorrectly reported as missing evidence.  ``transformer`` is the
# semantically preferred name for the current BrainUICL implementation: its
# one attention module is invoked repeatedly by the encoder rather than being
# three separately parameterized layer modules.  ``transformer_1`` remains a
# legacy spelling, not a claim that layer 1 was isolated.
METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "task.spectra.transformer_1.effective_rank": (
        "task.spectra.transformer_1.effective_rank",
        "task.spectra.transformer.effective_rank",
    ),
    "task.spectra.transformer.effective_rank": (
        "task.spectra.transformer.effective_rank",
        "task.spectra.transformer_1.effective_rank",
    ),
    "plasticity.acc_gain": (
        "plasticity.acc_gain",
        "task.plasticity.acc_gain",
    ),
    "task.plasticity.acc_gain": (
        "task.plasticity.acc_gain",
        "plasticity.acc_gain",
    ),
    "task.plasticity.fresh_gap": (
        "task.plasticity.fresh_gap",
        "plasticity.fresh_gap",
        "plasticity.fresh_gap_final",
        "task.plasticity.fresh_gap_final",
    ),
    "plasticity.fresh_gap": (
        "plasticity.fresh_gap",
        "task.plasticity.fresh_gap",
        "plasticity.fresh_gap_final",
        "task.plasticity.fresh_gap_final",
    ),
    "plasticity.fresh_gap_final": (
        "plasticity.fresh_gap_final",
        "task.plasticity.fresh_gap",
        "plasticity.fresh_gap",
        "task.plasticity.fresh_gap_final",
    ),
    "task.plasticity.fresh_gap_final": (
        "task.plasticity.fresh_gap_final",
        "task.plasticity.fresh_gap",
        "plasticity.fresh_gap",
        "plasticity.fresh_gap_final",
    ),
}


def metric_candidates(name: str) -> tuple[str, ...]:
    """Return equivalent metric spellings in preference order."""

    return METRIC_ALIASES.get(name, (name,))


def _resolve_metric_name(metrics: list[dict[str, Any]], name: str) -> str | None:
    available = {str(item.get("name")) for item in metrics if item.get("name")}
    return next((candidate for candidate in metric_candidates(name) if candidate in available), None)


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


def _pair_context_key(context: Any) -> str:
    """Keep stable task identity while ignoring measurement-only fields.

    Predictor and outcome are normally produced by different adapters and
    have different ``measurement_protocol``/``probe_budget`` values.  Those
    fields must remain in the stored metric context for auditability, but
    should not prevent an exact subject/split pairing.  Callers that need a
    stricter design can put a stable discriminator in ``task_context``.
    """

    value = context if isinstance(context, dict) else {}
    stable = {key: value[key] for key in PAIR_CONTEXT_KEYS if key in value}
    return json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _step_values(metrics: list[dict[str, Any]], name: str) -> dict[int, list[float]]:
    values: dict[int, list[float]] = defaultdict(list)
    candidates = set(metric_candidates(name))
    for metric in metrics:
        if metric.get("name") not in candidates or metric.get("step") is None:
            continue
        value = _finite(metric.get("value"))
        if value is not None:
            values[int(metric["step"])].append(value)
    return values


def _exact_values(metrics: list[dict[str, Any]], name: str) -> dict[tuple[int, str], float]:
    values: dict[tuple[int, str], list[float]] = defaultdict(list)
    candidates = set(metric_candidates(name))
    for metric in metrics:
        if metric.get("name") not in candidates or metric.get("step") is None:
            continue
        value = _finite(metric.get("value"))
        if value is not None:
            values[(int(metric["step"]), _pair_context_key(metric.get("context")))].append(value)
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
    predictor_resolved: set[str] = set()
    outcome_resolved: set[str] = set()
    for identity in identities:
        experiment_id = str(identity["experiment_id"])
        metrics = metrics_by_experiment.get(experiment_id, [])
        predictor_name = _resolve_metric_name(metrics, predictor)
        outcome_name = _resolve_metric_name(metrics, outcome)
        if predictor_name:
            predictor_resolved.add(predictor_name)
        if outcome_name:
            outcome_resolved.add(outcome_name)
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
        "predictor_requested": predictor,
        "outcome_requested": outcome,
        "predictor_resolved": sorted(predictor_resolved),
        "outcome_resolved": sorted(outcome_resolved),
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


# The correlation analyzer above is intentionally exploratory.  The gate below
# implements the stricter LoP evidence contract: the primary outcome is the
# fixed-budget fresh-vs-warm gap, and every comparison is made at the same
# stage grid and experimental design across independent seeds.
LOP_GATE_SCHEMA_VERSION = 1
LOP_GATE_DESIGN_FIELDS = (
    "dataset",
    "subject",
    "split",
    "task_order",
    "probe_budget",
    "optimizer",
    "learning_rate",
    "model_structure",
    "fresh_warm_protocol",
)
_LOP_GATE_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "dataset": ("dataset", "dataset_name"),
    "subject": ("subject", "subject_id"),
    "split": ("split", "evaluation_split"),
    "task_order": ("task_order", "task_sequence", "tasks", "task_context"),
    "probe_budget": ("probe_budget", "budget", "probe_steps"),
    "optimizer": ("optimizer", "optimizer_name"),
    "learning_rate": ("learning_rate", "lr"),
    "model_structure": ("model_structure", "model_config", "model", "architecture"),
    "fresh_warm_protocol": ("fresh_warm_protocol", "fresh_mode", "warm_fresh_mode", "measurement_protocol"),
}


def _json_value(value: Any) -> Any:
    """Return a deterministic, JSON-safe representation for design fields."""

    if isinstance(value, dict):
        return {str(key): _json_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _design_value(field: str, value: Any) -> Any:
    if field in {"subject", "split"}:
        return str(value)
    if isinstance(value, dict):
        if field == "dataset" and value.get("name") is not None:
            return str(value["name"])
        if field == "model_structure" and value.get("name") is not None:
            # Keep architecture-defining options when present, while ignoring
            # volatile checkpoint paths and device placement.
            ignored = {"checkpoint", "checkpoint_path", "device", "dtype"}
            return _json_value({key: item for key, item in value.items() if key not in ignored})
        return _json_value(value)
    if field == "probe_budget" and isinstance(value, (list, tuple)):
        numeric = [_finite(item) for item in value]
        if numeric and all(item is not None for item in numeric):
            return int(max(numeric))
    return _json_value(value)


def _containers(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    spec = experiment.get("spec") if isinstance(experiment.get("spec"), dict) else {}
    direct_metadata = experiment.get("metadata") if isinstance(experiment.get("metadata"), dict) else {}
    metadata = spec.get("metadata") if isinstance(spec.get("metadata"), dict) else {}
    config = experiment.get("config") if isinstance(experiment.get("config"), dict) else {}
    summary = experiment.get("summary") if isinstance(experiment.get("summary"), dict) else {}
    return [experiment, spec, direct_metadata, metadata, config, summary]


def _field_candidates(field: str, experiment: dict[str, Any], metrics: list[dict[str, Any]]) -> list[Any]:
    aliases = _LOP_GATE_FIELD_ALIASES[field]
    values: list[Any] = []
    for container in _containers(experiment):
        for alias in aliases:
            if alias in container and container[alias] is not None:
                values.append(_design_value(field, container[alias]))
    for metric in metrics:
        context = metric.get("context") if isinstance(metric.get("context"), dict) else {}
        for alias in aliases:
            if alias in context and context[alias] is not None:
                values.append(_design_value(field, context[alias]))
    # Preserve order for readable diagnostics, but remove duplicate JSON values.
    unique: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


def _gate_design(experiment: dict[str, Any], metrics: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str], dict[str, list[Any]]]:
    design: dict[str, Any] = {}
    missing: list[str] = []
    varied: dict[str, list[Any]] = {}
    for field in LOP_GATE_DESIGN_FIELDS:
        values = _field_candidates(field, experiment, metrics)
        if not values:
            missing.append(field)
            continue
        design[field] = values[0]
        if len(values) > 1:
            varied[field] = values
    return design, missing, varied


def _as_stage(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value.is_integer() else None
    if isinstance(value, str):
        text = value.strip()
        if text and (text.isdigit() or (text[0] in "+-" and text[1:].isdigit())):
            try:
                return int(text)
            except ValueError:
                return None
    return None


def _transition_key(metric: dict[str, Any]) -> int | tuple[int, int] | None:
    context = metric.get("context") if isinstance(metric.get("context"), dict) else {}
    source = next((context.get(key) for key in ("source_stage", "from_stage", "checkpoint_stage", "after_stage") if context.get(key) is not None), None)
    target = next((context.get(key) for key in ("target_stage", "to_stage", "outcome_stage", "target_task", "stage") if context.get(key) is not None), metric.get("step"))
    target_stage = _as_stage(target)
    if target_stage is None:
        return None
    source_stage = _as_stage(source)
    return (source_stage, target_stage) if source_stage is not None else target_stage


def _transition_label(value: int | tuple[int, int]) -> str:
    if isinstance(value, tuple):
        return f"{value[0]}->{value[1]}"
    return str(value)


def _transition_json(value: int | tuple[int, int]) -> int | list[int]:
    return list(value) if isinstance(value, tuple) else value


def _normalize_transition(value: Any) -> int | tuple[int, int] | None:
    if isinstance(value, str):
        text = value.strip().replace(":", "->")
        if "->" in text:
            left, right = text.split("->", 1)
            source, target = _as_stage(left.strip()), _as_stage(right.strip())
            return (source, target) if source is not None and target is not None else None
        return _as_stage(text)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        source, target = _as_stage(value[0]), _as_stage(value[1])
        return (source, target) if source is not None and target is not None else None
    return _as_stage(value)


def _transition_sort_key(value: int | tuple[int, int]) -> tuple[int, int, int]:
    if isinstance(value, tuple):
        return (1, value[0], value[1])
    return (0, value, value)


def _bootstrap_mean_interval(values: list[float], *, seed: int, repeats: int) -> list[float] | None:
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(repeats):
        sample = [values[rng.randrange(len(values))] for _ in values]
        samples.append(_mean(sample))
    samples.sort()
    return [round(_percentile(samples, 0.025), 8), round(_percentile(samples, 0.975), 8)]


def evaluate_lop_gate(
    experiments: list[dict[str, Any]],
    metrics_by_experiment: dict[str, list[dict[str, Any]]],
    *,
    outcome: str = DEFAULT_LOP_OUTCOME,
    required_stages: Iterable[int] | None = None,
    required_transitions: Iterable[Any] | None = None,
    minimum_seeds: int = SCIENTIFIC_MINIMUM_SEEDS,
    minimum_stages: int = 2,
    bootstrap_repeats: int = 2000,
    bootstrap_seed: int = 20260830,
    direction: str = "fresh-better",
) -> dict[str, Any]:
    """Evaluate whether a fixed-budget fresh-gap trajectory is LoP-ready.

    ``candidate`` means the preregistered evidence checks passed; it is still
    not a scientific conclusion.  Replay/retention rows are reported as a
    separate inventory and are never used as the primary outcome.
    """

    if direction not in {"fresh-better", "positive", "gap-positive"}:
        raise ValueError("direction must be fresh-better, positive or gap-positive")
    minimum_seeds = min(10_000, max(SCIENTIFIC_MINIMUM_SEEDS, int(minimum_seeds)))
    minimum_stages = min(10_000, max(2, int(minimum_stages)))
    bootstrap_repeats = min(10_000, max(100, int(bootstrap_repeats)))
    requested_stages = {_as_stage(item) for item in (required_stages or [])}
    requested_stages.discard(None)
    requested_transition_values = {
        item for item in (_normalize_transition(value) for value in (required_transitions or [])) if item is not None
    }
    allowed_outcomes = set(metric_candidates(DEFAULT_LOP_OUTCOME))
    outcome_is_primary = outcome in allowed_outcomes
    identities: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []
    design_by_experiment: dict[str, dict[str, Any]] = {}
    missing_design_by_experiment: dict[str, list[str]] = {}
    varied_design_by_experiment: dict[str, dict[str, list[Any]]] = {}
    role_violations: list[dict[str, Any]] = []
    inferred_roles: list[dict[str, Any]] = []
    retention_inventory: list[dict[str, Any]] = []
    missing_seed_experiments: list[str] = []

    for experiment in sorted(experiments, key=lambda item: str(item.get("experiment_id") or "")):
        spec = experiment.get("spec") if isinstance(experiment.get("spec"), dict) else {}
        metadata = spec.get("metadata") if isinstance(spec.get("metadata"), dict) else {}
        experiment_id = str(experiment.get("experiment_id") or spec.get("experiment_id") or "")
        seed = experiment.get("seed", spec.get("seed"))
        identities.append({
            "experiment_id": experiment_id,
            "seed": seed,
            "dataset": experiment.get("dataset", spec.get("dataset")),
            "subject": experiment.get("subject", metadata.get("subject")),
            "split": experiment.get("split", metadata.get("split")),
            "method": experiment.get("method", spec.get("method")),
        })
        if not isinstance(seed, int) or isinstance(seed, bool):
            missing_seed_experiments.append(experiment_id)
        metrics = metrics_by_experiment.get(experiment_id, [])
        design, missing, varied = _gate_design(experiment, metrics)
        design_by_experiment[experiment_id] = design
        if missing:
            missing_design_by_experiment[experiment_id] = missing
        if varied:
            varied_design_by_experiment[experiment_id] = varied
        candidates = set(metric_candidates(outcome))
        for metric in metrics:
            name = metric.get("name")
            context = metric.get("context") if isinstance(metric.get("context"), dict) else {}
            role = context.get("metric_role", metric.get("metric_role"))
            if role == "retention" or "retention" in str(name).lower() or "forgetting" in str(name).lower():
                if isinstance(name, str):
                    retention_inventory.append({"experiment_id": experiment_id, "name": name, "step": metric.get("step")})
            if name not in candidates:
                continue
            if role is None:
                inferred_roles.append({"experiment_id": experiment_id, "name": name})
            elif outcome_is_primary and role != "outcome":
                role_violations.append({"experiment_id": experiment_id, "name": name, "metric_role": role})
            value = _finite(metric.get("value"))
            transition = _transition_key(metric)
            if value is not None and transition is not None:
                value_rows.append({"experiment_id": experiment_id, "seed": seed, "transition": transition, "value": value, "context": context})

    reasons: list[str] = []
    status = "candidate"
    if not outcome_is_primary:
        status = "blocked-invalid-outcome"
        reasons.append(f"primary LoP outcome must be fresh-gap; received {outcome!r}")
    if role_violations:
        status = "blocked-metric-role"
        reasons.append("fresh-gap rows must have metric_role=outcome; retention/diagnostic rows cannot be substituted")

    valid_seed_values = [item["seed"] for item in identities if isinstance(item["seed"], int) and not isinstance(item["seed"], bool)]
    seeds = sorted({item["seed"] for item in value_rows if isinstance(item["seed"], int) and not isinstance(item["seed"], bool)})
    duplicate_seeds = len(valid_seed_values) != len(set(valid_seed_values))
    if duplicate_seeds:
        status = "blocked-duplicate-seeds"
        reasons.append("each seed must identify one independent trajectory; duplicate seed runs are not independent evidence")
    if len(seeds) < minimum_seeds:
        if status == "candidate":
            status = "insufficient-seeds"
        reasons.append(f"usable seeds={len(seeds)} is below minimum_seeds={minimum_seeds}")
    if missing_seed_experiments:
        reasons.append(f"experiments without an integer real seed: {sorted(missing_seed_experiments)}")
    if missing_design_by_experiment:
        if status == "candidate":
            status = "blocked-missing-design"
        reasons.append("all trajectories must declare dataset, subject, split, task order, probe budget, optimizer, learning rate and model structure")
    if varied_design_by_experiment:
        if status == "candidate":
            status = "blocked-varying-design"
        reasons.append("a trajectory contains conflicting fixed-budget design metadata")
    design_signatures = {
        experiment_id: json.dumps(design, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for experiment_id, design in design_by_experiment.items()
    }
    comparable_design = len(set(design_signatures.values())) <= 1 and not missing_design_by_experiment and not varied_design_by_experiment
    if not comparable_design and not (missing_design_by_experiment or varied_design_by_experiment):
        if status == "candidate":
            status = "blocked-incomparable-design"
        reasons.append("dataset, subject, split, task order, probe budget, optimizer, learning rate and model structure must match across seeds")

    by_experiment_grid: dict[str, set[int | tuple[int, int]]] = defaultdict(set)
    by_transition_seed_values: dict[int | tuple[int, int], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in value_rows:
        by_experiment_grid[row["experiment_id"]].add(row["transition"])
        if isinstance(row["seed"], int) and not isinstance(row["seed"], bool):
            by_transition_seed_values[row["transition"]][row["seed"]].append(row["value"])
    grids = {experiment_id: sorted(grid, key=_transition_sort_key) for experiment_id, grid in by_experiment_grid.items()}
    nonempty_grids = {tuple(grid) for grid in grids.values() if grid}
    stage_grid = next(iter(nonempty_grids), tuple())
    stages_consistent = len(nonempty_grids) <= 1 and len(grids) == len(identities)
    if not stages_consistent:
        if status == "candidate":
            status = "blocked-incomparable-stages"
        reasons.append("all seeds must expose the same ordered stage/transition grid")
    expected_grid: set[int | tuple[int, int]] = set(stage_grid)
    if requested_stages:
        expected_grid = {int(item) for item in requested_stages}
    if requested_transition_values:
        expected_grid = set(requested_transition_values)
    if expected_grid and any(set(grid) != expected_grid for grid in grids.values()):
        if status == "candidate":
            status = "blocked-incomparable-stages"
        reasons.append("observed stage/transition grid does not match the required grid")
    if not expected_grid:
        if status == "candidate":
            status = "blocked-missing-outcome"
        reasons.append("no finite fresh-gap rows with a usable stage or transition were found")
    missing_stage_experiments = sorted(set(identity["experiment_id"] for identity in identities) - set(grids))
    if missing_stage_experiments:
        reasons.append(f"experiments without a usable stage/transition: {missing_stage_experiments}")
    if len(expected_grid) < minimum_stages:
        if status == "candidate":
            status = "insufficient-stages"
        reasons.append(f"stage_count={len(expected_grid)} is below minimum_stages={minimum_stages}")

    stage_summaries: list[dict[str, Any]] = []
    all_positive = True
    mixed_direction = False
    confidence_supported = True
    for index, transition in enumerate(sorted(expected_grid, key=_transition_sort_key)):
        seed_values = {seed: _mean(values) for seed, values in by_transition_seed_values.get(transition, {}).items()}
        ordered_values = [seed_values[seed] for seed in sorted(seed_values)]
        missing_seeds = sorted(set(seeds) - set(seed_values))
        positive_count = sum(value > 0 for value in ordered_values)
        negative_count = sum(value < 0 for value in ordered_values)
        zero_count = sum(value == 0 for value in ordered_values)
        all_positive = all_positive and bool(ordered_values) and positive_count == len(ordered_values)
        mixed_direction = mixed_direction or (positive_count > 0 and (negative_count > 0 or zero_count > 0)) or (negative_count > 0 and zero_count > 0)
        ci = _bootstrap_mean_interval(ordered_values, seed=bootstrap_seed + index, repeats=bootstrap_repeats)
        if ci is None or ci[0] <= 0:
            confidence_supported = False
        stage_summaries.append({
            "transition": _transition_label(transition),
            "transition_key": _transition_json(transition),
            "source_stage": transition[0] if isinstance(transition, tuple) else None,
            "outcome_stage": transition[1] if isinstance(transition, tuple) else transition,
            "seed_count": len(ordered_values),
            "missing_seeds": missing_seeds,
            "seed_values": {str(seed): round(value, 8) for seed, value in sorted(seed_values.items())},
            "mean": round(_mean(ordered_values), 8) if ordered_values else None,
            "median": round(_percentile(sorted(ordered_values), 0.5), 8) if ordered_values else None,
            "std": round(math.sqrt(sum((value - _mean(ordered_values)) ** 2 for value in ordered_values) / len(ordered_values)), 8) if ordered_values else None,
            "positive_seed_count": positive_count,
            "negative_seed_count": negative_count,
            "zero_seed_count": zero_count,
            "bootstrap_ci95": ci,
            "direction_supported": bool(ordered_values) and positive_count == len(ordered_values) and ci is not None and ci[0] > 0,
        })
    if role_violations == [] and outcome_is_primary and expected_grid and not all_positive:
        if mixed_direction:
            if status == "candidate":
                status = "blocked-inconsistent-direction"
            reasons.append("fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction")
        elif status == "candidate":
            status = "insufficient-direction"
            reasons.append("at least one stage has a zero or non-positive seed-level fresh gap")
    if status == "candidate" and not confidence_supported:
        status = "insufficient-direction"
        reasons.append("seed-cluster 95% CI does not stay strictly above zero for every stage")

    result = {
        "schema_version": LOP_GATE_SCHEMA_VERSION,
        "analysis": "lop-requirement-gate-v1",
        "status": status,
        "scientific_conclusion_allowed": False,
        "outcome": outcome,
        "outcome_role": "outcome" if outcome_is_primary else "invalid",
        "direction": "fresh-better (fresh_gap > 0)",
        "minimum_seeds": minimum_seeds,
        "minimum_stages": minimum_stages,
        "bootstrap": {"method": "seed-cluster-mean-percentile", "repeats": bootstrap_repeats, "seed": bootstrap_seed},
        "experiment_count": len(identities),
        "seed_count": len(seeds),
        "seeds": seeds,
        "duplicate_seeds": duplicate_seeds,
        "missing_seed_experiments": sorted(missing_seed_experiments),
        "missing_stage_experiments": missing_stage_experiments,
        "stage_count": len(expected_grid),
        "required_stages": sorted(requested_stages) if requested_stages else [],
        "required_transitions": [_transition_json(item) for item in sorted(requested_transition_values, key=_transition_sort_key)],
        "required_transition_labels": [_transition_label(item) for item in sorted(requested_transition_values, key=_transition_sort_key)],
        "stage_grid": [_transition_json(item) for item in sorted(expected_grid, key=_transition_sort_key)],
        "stage_grid_labels": [_transition_label(item) for item in sorted(expected_grid, key=_transition_sort_key)],
        "stage_grid_by_experiment": {key: [_transition_json(item) for item in value] for key, value in sorted(grids.items())},
        "stage_grid_labels_by_experiment": {key: [_transition_label(item) for item in value] for key, value in sorted(grids.items())},
        "stages_consistent": stages_consistent,
        "design_consistent": comparable_design,
        "design_by_experiment": design_by_experiment,
        "missing_design_by_experiment": missing_design_by_experiment,
        "varied_design_by_experiment": varied_design_by_experiment,
        "role_violations": role_violations,
        "inferred_outcome_roles": inferred_roles,
        "retention_separate": True,
        "retention_inventory": retention_inventory,
        "stage_summaries": stage_summaries,
        "positive_stage_count": sum(item["direction_supported"] for item in stage_summaries),
        "direction_supported": bool(stage_summaries) and all(item["direction_supported"] for item in stage_summaries),
        "reasons": reasons,
        "interpretation": "candidate is a requirement-gate result only; retention/forgetting and predictor metrics are separate evidence and cannot establish LoP causality",
        "experiments": identities,
    }
    result["gate_digest"] = hashlib.sha256(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return result


__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "DEFAULT_PREDICTOR",
    "DEFAULT_OUTCOME",
    "DEFAULT_LOP_OUTCOME",
    "SCIENTIFIC_MINIMUM_SEEDS",
    "analyze_lop",
    "evaluate_lop_gate",
    "metric_candidates",
]
