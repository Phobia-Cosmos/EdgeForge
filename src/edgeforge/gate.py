"""Validation helpers for the immutable model capability gate."""

from __future__ import annotations

import math
import uuid
from typing import Any


GATE_OPERATORS = {">=", ">", "<=", "<"}


def _identifier(value: Any, field: str, *, prefix: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        return f"{prefix}-{uuid.uuid4().hex}"
    if len(normalized) > 160 or any(character.isspace() for character in normalized):
        raise ValueError(f"{field} must be a non-empty identifier without whitespace")
    return normalized


def normalize_model(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("model registration must be an object")
    name = str(data.get("name") or "").strip()
    workload = str(data.get("workload") or "").strip()
    protocol = str(data.get("protocol") or "").strip()
    comparison_group = str(data.get("comparison_group") or "").strip()
    source_experiment_id = str(data.get("source_experiment_id") or "").strip()
    descriptor = data.get("descriptor") or {}
    if not all((name, workload, protocol, comparison_group, source_experiment_id)):
        raise ValueError(
            "model name, workload, protocol, comparison_group and source_experiment_id are required"
        )
    if not isinstance(descriptor, dict):
        raise ValueError("model descriptor must be an object")
    checkpoint_digest = data.get("checkpoint_digest")
    if checkpoint_digest is not None:
        checkpoint_digest = str(checkpoint_digest).lower()
        if len(checkpoint_digest) != 64 or any(character not in "0123456789abcdef" for character in checkpoint_digest):
            raise ValueError("checkpoint_digest must be a SHA-256 hex digest")
    status = str(data.get("status") or "candidate")
    if status != "candidate":
        raise ValueError("new models must start in candidate status")
    return {
        "id": _identifier(data.get("id"), "model id", prefix="model"),
        "name": name,
        "workload": workload,
        "protocol": protocol,
        "comparison_group": comparison_group,
        "source_experiment_id": source_experiment_id,
        "checkpoint_digest": checkpoint_digest,
        "descriptor": descriptor,
        "status": status,
    }


def normalize_policy(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("gate policy must be an object")
    version = str(data.get("version") or "").strip()
    workload = str(data.get("workload") or "").strip()
    protocol = str(data.get("protocol") or "").strip()
    comparison_group = str(data.get("comparison_group") or "").strip()
    rules = data.get("rules")
    if not all((version, workload, protocol, comparison_group)):
        raise ValueError("policy version, workload, protocol and comparison_group are required")
    if not isinstance(rules, list) or not rules or len(rules) > 100:
        raise ValueError("policy rules must be a non-empty array with at most 100 entries")
    normalized_rules = []
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ValueError(f"policy rule {index} must be an object")
        metric = str(rule.get("metric") or "").strip()
        operator = str(rule.get("operator") or "")
        threshold = rule.get("threshold")
        step = rule.get("step")
        namespace = rule.get("namespace")
        if not metric:
            raise ValueError(f"policy rule {index} requires metric")
        if operator not in GATE_OPERATORS:
            raise ValueError(f"policy rule {index} uses unsupported operator: {operator}")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
            raise ValueError(f"policy rule {index} threshold must be a finite number")
        if step is not None and (isinstance(step, bool) or not isinstance(step, int)):
            raise ValueError(f"policy rule {index} step must be an integer or null")
        if namespace is not None and not str(namespace).strip():
            raise ValueError(f"policy rule {index} namespace must be non-empty when present")
        normalized = {
            "metric": metric,
            "operator": operator,
            "threshold": float(threshold),
            "step": step,
        }
        if namespace is not None:
            normalized["namespace"] = str(namespace).strip()
        normalized_rules.append(normalized)
    return {
        "id": _identifier(data.get("id"), "policy id", prefix="policy"),
        "version": version,
        "workload": workload,
        "protocol": protocol,
        "comparison_group": comparison_group,
        "rules": normalized_rules,
    }


def compare(value: float, operator: str, threshold: float) -> bool:
    if operator == ">=":
        return value >= threshold
    if operator == ">":
        return value > threshold
    if operator == "<=":
        return value <= threshold
    if operator == "<":
        return value < threshold
    raise ValueError(f"unsupported gate operator: {operator}")
