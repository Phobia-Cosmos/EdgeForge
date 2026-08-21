"""Local, read-only LoP evidence audit for imported RA-EEG result files."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from edgeforge.experiment import normalize_raeeg_metrics
from edgeforge.lop_analysis import DEFAULT_OUTCOME, DEFAULT_PREDICTOR, analyze_lop


MAX_AUDIT_RESULT_BYTES = 16 * 1024 * 1024


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        if path.stat().st_size > MAX_AUDIT_RESULT_BYTES:
            raise ValueError(f"result exceeds {MAX_AUDIT_RESULT_BYTES} bytes: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON result: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"result must be a JSON object: {path}")
    return value


def _normalized_metrics(raw: dict[str, Any]) -> list[dict[str, Any]]:
    direct = raw.get("metrics")
    if isinstance(direct, list) and all(
        isinstance(item, dict) and isinstance(item.get("name"), str)
        for item in direct
    ):
        return direct
    metrics, _summary = normalize_raeeg_metrics(raw)
    return metrics


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _metric_names(metrics: list[dict[str, Any]]) -> set[str]:
    return {str(item.get("name")) for item in metrics if item.get("name")}


def _metric_steps(metrics: list[dict[str, Any]], name: str) -> list[int]:
    return sorted({int(item["step"]) for item in metrics if item.get("name") == name and item.get("step") is not None})


def _entry_identity(entry: dict[str, Any], source_digest: str | None = None) -> dict[str, Any]:
    metadata = entry.get("metadata") or {}
    return {
        "experiment_id": str(entry.get("experiment_id") or ""),
        "workload": entry.get("workload"),
        "protocol": entry.get("protocol"),
        "method": entry.get("method"),
        "seed": entry.get("seed"),
        "spec": {"metadata": metadata},
        "artifact_digest": source_digest or metadata.get("source_digest"),
        "source_digest": source_digest or metadata.get("source_digest"),
    }


def audit_catalog(
    catalog_path: str | Path,
    *,
    predictor: str = DEFAULT_PREDICTOR,
    outcome: str = DEFAULT_OUTCOME,
    context_policy: str = "aggregate-step",
    bootstrap_repeats: int = 1000,
    bootstrap_seed: int = 20260821,
    minimum_pairs: int = 3,
    minimum_seeds: int = 3,
    methods: list[str] | None = None,
) -> dict[str, Any]:
    """Audit a catalog without uploading or mutating its source results."""
    path = Path(catalog_path).resolve()
    catalog = _load_json(path)
    if catalog.get("schema_version") != 1 or not isinstance(catalog.get("experiments"), list):
        raise ValueError("unsupported catalog schema; expected schema_version=1 and experiments[]")
    root = Path(str(catalog.get("worker_work_root") or path.parent)).expanduser().resolve()
    records: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str, str], list[tuple[dict[str, Any], list[dict[str, Any]]]]] = defaultdict(list)
    defaults = catalog.get("defaults") or {}
    method_filter = {str(item).strip() for item in (methods or []) if str(item).strip()}
    for raw_entry in catalog["experiments"]:
        entry = _merge(defaults, raw_entry) if isinstance(raw_entry, dict) else {}
        if not isinstance(entry, dict):
            continue
        if method_filter and str(entry.get("method") or "") not in method_filter:
            continue
        relative = str((entry.get("runner") or {}).get("result_path") or "")
        source = (root / relative).resolve() if relative else Path("/")
        record: dict[str, Any] = {
            "experiment_id": entry.get("experiment_id"),
            "method": entry.get("method"),
            "protocol": entry.get("protocol"),
            "comparison_group": (entry.get("metadata") or {}).get("comparison_group"),
            "seed": entry.get("seed"),
            "replay": (entry.get("metadata") or {}).get("replay"),
            "source_path": relative,
            "source_exists": source.is_file() and source.is_relative_to(root),
            "status": "missing-source",
            "predictor_count": 0,
            "outcome_count": 0,
            "predictor_steps": [],
            "outcome_steps": [],
        }
        if record["source_exists"]:
            try:
                raw = _load_json(source)
                metrics = _normalized_metrics(raw)
                source_digest = _digest(source)
            except (OSError, ValueError) as error:
                record["status"] = "invalid-source"
                record["error"] = str(error)
                records.append(record)
                continue
            names = _metric_names(metrics)
            record.update({
                "source_digest": source_digest,
                "metric_count": len(metrics),
                "predictor_count": sum(item.get("name") == predictor for item in metrics),
                "outcome_count": sum(item.get("name") == outcome for item in metrics),
                "predictor_steps": _metric_steps(metrics, predictor),
                "outcome_steps": _metric_steps(metrics, outcome),
                "has_predictor": predictor in names,
                "has_outcome": outcome in names,
            })
            if not record["has_predictor"]:
                record["status"] = "missing-predictor"
            elif not record["has_outcome"]:
                record["status"] = "missing-outcome"
            else:
                record["status"] = "candidate"
            identity = _entry_identity(entry, record["source_digest"])
            key = (
                str(entry.get("workload") or ""),
                str(entry.get("protocol") or ""),
                str((entry.get("metadata") or {}).get("comparison_group") or ""),
                str(entry.get("method") or ""),
            )
            groups[key].append((identity, metrics))
        records.append(record)

    analyses: list[dict[str, Any]] = []
    records_by_experiment = {str(record["experiment_id"]): record for record in records}
    for key in sorted(groups):
        experiments_and_metrics = groups[key]
        group_records = [records_by_experiment[item[0]["experiment_id"]] for item in experiments_and_metrics]
        evidence_status = "candidate"
        if any(item["status"] == "missing-predictor" for item in group_records):
            evidence_status = "missing-predictor"
        elif any(item["status"] == "missing-outcome" for item in group_records):
            evidence_status = "missing-outcome"
        result = analyze_lop(
            [item[0] for item in experiments_and_metrics],
            {item[0]["experiment_id"]: item[1] for item in experiments_and_metrics},
            predictor=predictor,
            outcome=outcome,
            context_policy=context_policy,
            bootstrap_repeats=bootstrap_repeats,
            bootstrap_seed=bootstrap_seed,
            minimum_pairs=minimum_pairs,
            minimum_seeds=minimum_seeds,
        )
        analyses.append({
            "workload": key[0],
            "protocol": key[1],
            "comparison_group": key[2],
            "method": key[3],
            "experiment_ids": [item[0]["experiment_id"] for item in experiments_and_metrics],
            "evidence_status": evidence_status,
            "result": result,
        })
        for record in records:
            if record["experiment_id"] in result["contributing_experiments"] and record["status"] == "candidate":
                record["status"] = "analyzed" if result["status"] == "ok" else "analysis-blocked"

    status_counts: dict[str, int] = defaultdict(int)
    for record in records:
        status_counts[str(record["status"])] += 1
    method_summary: dict[str, dict[str, Any]] = {}
    for record in records:
        method = str(record.get("method") or "unknown")
        summary = method_summary.setdefault(method, {
            "record_count": 0,
            "source_missing": 0,
            "predictor_available": 0,
            "outcome_available": 0,
            "replay_values": [],
            "statuses": [],
        })
        summary["record_count"] += 1
        summary["source_missing"] += int(not record["source_exists"])
        summary["predictor_available"] += int(record.get("has_predictor", False))
        summary["outcome_available"] += int(record.get("has_outcome", False))
        if record.get("replay") not in summary["replay_values"]:
            summary["replay_values"].append(record.get("replay"))
        if record["status"] not in summary["statuses"]:
            summary["statuses"].append(record["status"])
    for method, summary in method_summary.items():
        summary["replay_values"].sort(key=lambda value: str(value))
        summary["statuses"].sort()
    return {
        "schema_version": 1,
        "audit": "raeeg-lop-evidence-v1",
        "catalog": str(path),
        "worker_work_root": str(root),
        "predictor": predictor,
        "predictor_role": "er" if predictor == DEFAULT_PREDICTOR else "custom",
        "outcome": outcome,
        "analysis_scope": "lop-er-plasticity" if predictor == DEFAULT_PREDICTOR else "exploratory-custom-association",
        "method_filter": sorted(method_filter),
        "context_policy": context_policy,
        "minimum_seeds": max(3, int(minimum_seeds)),
        "scientific_conclusion_allowed": False,
        "interpretation": (
            "ER-to-plasticity evidence audit; analysis remains descriptive and non-causal"
            if predictor == DEFAULT_PREDICTOR
            else "custom predictor association audit; not an ER-based LoP analysis and not causal"
        ),
        "record_count": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "method_summary": dict(sorted(method_summary.items())),
        "records": records,
        "analyses": analyses,
    }
