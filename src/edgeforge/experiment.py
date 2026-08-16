"""Versioned experiment contracts and RA-EEG metric normalization."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any


IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
MAX_METRICS = 20_000


def _object_with_name(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("name"), str) or not value["name"].strip():
        raise ValueError(f"experiment {field} must be an object with a non-empty name")
    return dict(value)


@dataclass(frozen=True)
class ExperimentSpec:
    schema_version: int
    experiment_id: str
    workload: str
    dataset: dict[str, Any]
    model: dict[str, Any]
    protocol: str
    method: str
    seed: int
    runner: dict[str, Any]
    metadata: dict[str, Any]

    @classmethod
    def from_payload(cls, value: Any) -> "ExperimentSpec":
        if not isinstance(value, dict):
            raise ValueError("experiment spec must be an object")
        schema_version = int(value.get("schema_version") or 1)
        if schema_version != 1:
            raise ValueError("unsupported experiment schema_version")
        experiment_id = str(value.get("experiment_id") or "")
        workload = str(value.get("workload") or "")
        protocol = str(value.get("protocol") or "")
        method = str(value.get("method") or "")
        for field, item in (
            ("experiment_id", experiment_id),
            ("workload", workload),
            ("protocol", protocol),
            ("method", method),
        ):
            if not IDENTIFIER.fullmatch(item):
                raise ValueError(f"experiment {field} must match {IDENTIFIER.pattern}")
        seed = value.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("experiment seed must be an integer")
        runner = value.get("runner")
        if not isinstance(runner, dict):
            raise ValueError("experiment runner must be an object")
        runner = dict(runner)
        mode = str(runner.get("mode") or ("command" if runner.get("argv") else "import"))
        if mode not in {"command", "import"}:
            raise ValueError("experiment runner mode must be command or import")
        runner["mode"] = mode
        if mode == "command":
            argv = runner.get("argv")
            if not isinstance(argv, list) or not argv or not all(isinstance(item, str) and item for item in argv):
                raise ValueError("command experiment runner requires a non-empty argv string array")
        result_path = runner.get("result_path")
        if not isinstance(result_path, str) or not result_path:
            raise ValueError("experiment runner result_path is required")
        adapter = str(runner.get("adapter") or "raeeg-metrics-v1")
        if adapter not in {"raeeg-metrics-v1", "edgeforge-bundle-v1"}:
            raise ValueError("unsupported experiment result adapter")
        runner["adapter"] = adapter
        metadata = value.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("experiment metadata must be an object")
        return cls(
            schema_version=schema_version,
            experiment_id=experiment_id,
            workload=workload,
            dataset=_object_with_name(value.get("dataset"), "dataset"),
            model=_object_with_name(value.get("model"), "model"),
            protocol=protocol,
            method=method,
            seed=seed,
            runner=runner,
            metadata=dict(metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "workload": self.workload,
            "dataset": self.dataset,
            "model": self.model,
            "protocol": self.protocol,
            "method": self.method,
            "seed": self.seed,
            "runner": self.runner,
            "metadata": self.metadata,
        }


def _unit(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("acc", "mf1", "bwt", "forget", "coverage", "rate", "fraction", "gain", "pearson")):
        return "ratio"
    if "latency" in lowered or lowered.endswith("_ms"):
        return "ms"
    if "bytes" in lowered or "memory_size" in lowered or lowered.endswith("_count"):
        return "count"
    return "scalar"


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


class _MetricCollector:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self.seen: set[tuple[str, int | None, str]] = set()

    def add(self, name: str, value: Any, *, step: int | None = None, context: dict[str, Any] | None = None) -> None:
        number = _numeric(value)
        if number is None or len(self.items) >= MAX_METRICS:
            return
        clean_name = name.strip(".")[:240]
        context = context or {}
        context_key = json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        key = (clean_name, step, context_key)
        if not clean_name or key in self.seen:
            return
        self.seen.add(key)
        self.items.append(
            {
                "namespace": "raeeg.research",
                "name": clean_name,
                "value": number,
                "step": step,
                "unit": _unit(clean_name),
                "context": context,
            }
        )

    def flatten(self, prefix: str, value: Any, *, step: int | None = None, context: dict[str, Any] | None = None) -> None:
        if _numeric(value) is not None:
            self.add(prefix, value, step=step, context=context)
            return
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(key, str):
                    self.flatten(f"{prefix}.{key}" if prefix else key, item, step=step, context=context)
            return
        if isinstance(value, list) and all(_numeric(item) is not None for item in value):
            for index, item in enumerate(value, start=1):
                self.add(prefix, item, step=index, context=context)


def _task_rows(raw: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        raw.get("tasks"),
        (raw.get("performance") or {}).get("tasks") if isinstance(raw.get("performance"), dict) else None,
        (raw.get("performance") or {}).get("results") if isinstance(raw.get("performance"), dict) else None,
    ]
    for value in candidates:
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return value
    return []


def normalize_raeeg_metrics(raw: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(raw, dict):
        raise ValueError("RA-EEG result must be a JSON object")
    collector = _MetricCollector()
    summary = raw.get("summary") or {}
    if isinstance(summary, dict):
        collector.flatten("summary", summary)

    performance = raw.get("performance")
    if isinstance(performance, dict):
        for key in ("stability", "plasticity", "forgetting", "spectrum", "spectra", "weight_norms"):
            if key in performance:
                collector.flatten(f"performance.{key}", performance[key])

    for index, task in enumerate(_task_rows(raw), start=1):
        step_value = task.get("task", task.get("task_id", index))
        try:
            step = int(step_value)
        except (TypeError, ValueError):
            step = index
        context = {}
        if task.get("subject") is not None:
            context["subject"] = str(task["subject"])
        for key in (
            "plasticity",
            "current_before",
            "current_after",
            "before",
            "after",
            "old_generalization_after",
            "all_seen_mean_after",
            "pseudo_labels",
            "importance",
            "spectrum",
            "spectra",
            "weight_norms",
        ):
            if key in task:
                collector.flatten(f"task.{key}", task[key], step=step, context=context)
        before = task.get("current_before") or task.get("before")
        after = task.get("current_after") or task.get("after")
        if isinstance(before, dict) and isinstance(after, dict):
            for metric in ("acc", "mf1"):
                left = _numeric(before.get(metric))
                right = _numeric(after.get(metric))
                if left is not None and right is not None:
                    collector.add(f"plasticity.{metric}_gain", right - left, step=step, context=context)

    return collector.items, summary if isinstance(summary, dict) else {}


def build_experiment_bundle(
    spec: ExperimentSpec,
    raw: dict[str, Any],
    *,
    source_path: str,
    source_bytes: bytes,
    environment: dict[str, Any],
) -> dict[str, Any]:
    adapter = spec.runner["adapter"]
    if adapter == "edgeforge-bundle-v1":
        if not isinstance(raw.get("metrics"), list):
            raise ValueError("EdgeForge bundle result requires a metrics array")
        metrics = raw["metrics"]
        summary = raw.get("summary") or {}
    else:
        metrics, summary = normalize_raeeg_metrics(raw)
    return {
        "schema_version": 1,
        "experiment_id": spec.experiment_id,
        "workload": spec.workload,
        "spec": spec.to_dict(),
        "metrics": metrics,
        "summary": summary,
        "source_result": {
            "path": source_path,
            "algorithm": "sha256",
            "digest": hashlib.sha256(source_bytes).hexdigest(),
            "size_bytes": len(source_bytes),
        },
        "environment": environment,
    }


def bundle_artifact_upload(bundle: dict[str, Any]) -> dict[str, Any]:
    content = json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "name": f"{bundle['experiment_id']}-bundle.json",
        "kind": "experiment-bundle",
        "media_type": "application/json",
        "metadata": {
            "experiment_id": bundle["experiment_id"],
            "workload": bundle["workload"],
            "schema_version": bundle["schema_version"],
        },
        "content_base64": base64.b64encode(content).decode("ascii"),
    }
