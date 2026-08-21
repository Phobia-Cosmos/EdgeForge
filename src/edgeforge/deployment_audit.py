"""Read-only deployment readiness audit for a model manifest and target probe.

The audit joins three different kinds of evidence without conflating them:

* the model manifest declares the requested backend and target;
* a target probe records what the Worker explicitly advertised;
* a model run records that the manifest actually loaded and passed correctness.

Presence of a SoC string, driver file, or inferred accelerator is never enough
to mark a deployment ready.  A missing model run therefore remains blocked,
which makes this module suitable for a release-matrix/CI gate.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from edgeforge.model_pipeline import normalize_model_manifest


AUDIT_VERSION = "edgeforge-deployment-audit-v1"
PROBE_SCHEMA_VERSION = 1


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _list_of_strings(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{name} must be an array of non-empty strings")
    return list(value)


def _check(name: str, status: str, reason: str, **details: Any) -> dict[str, Any]:
    return {"name": name, "status": status, "reason": reason, "details": details}


def _probe_digest(probe: dict[str, Any]) -> str:
    payload = {key: value for key, value in probe.items() if key != "generated_at" and key != "probe_digest"}
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _normalise_runs(model_runs: Iterable[dict[str, Any]] | dict[str, Any] | None) -> list[dict[str, Any]]:
    if model_runs is None:
        return []
    if isinstance(model_runs, dict):
        model_runs = model_runs.get("model_runs")
    if not isinstance(model_runs, list):
        raise ValueError("model_runs must be an array or an object containing model_runs")
    result = []
    for index, value in enumerate(model_runs):
        item = _object(value, f"model_runs[{index}]")
        result.append(item)
    return result


def audit_deployment(
    manifest_payload: dict[str, Any],
    probe_payload: dict[str, Any],
    model_runs: Iterable[dict[str, Any]] | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit whether one manifest has portable, correctness-backed target evidence.

    The returned status is ``ready`` only when every required check passes,
    including at least one successful model run with ``correctness=true`` that
    matches the canonical manifest digest, backend, dataset, and target
    architecture.  No performance or scientific claim is inferred here.
    """

    manifest = normalize_model_manifest(manifest_payload)
    probe = _object(probe_payload, "target probe")
    if int(probe.get("schema_version", -1)) != PROBE_SCHEMA_VERSION:
        raise ValueError(f"unsupported target probe schema_version: {probe.get('schema_version')}")
    capabilities = _object(probe.get("capabilities"), "target probe capabilities")
    claims = _object(probe.get("backend_claims"), "target probe backend_claims")
    target = manifest.get("target") or {}
    backend = str((manifest.get("compiler") or {}).get("backend") or "")
    target_architecture = str(target.get("architecture") or "")
    probe_architecture = str(capabilities.get("architecture") or "")
    advertised = _list_of_strings(claims.get("advertised"), "target probe backend_claims.advertised")
    inferred = _list_of_strings(claims.get("inferred"), "target probe backend_claims.inferred")
    accelerators = _list_of_strings(capabilities.get("accelerators"), "target probe capabilities.accelerators")
    checks: list[dict[str, Any]] = []

    recorded_digest = probe.get("probe_digest")
    if isinstance(recorded_digest, str) and recorded_digest:
        expected_digest = _probe_digest(probe)
        checks.append(
            _check(
                "probe-integrity",
                "pass" if recorded_digest == expected_digest else "blocked",
                "probe digest matches its content" if recorded_digest == expected_digest else "probe digest does not match its content",
                recorded_digest=recorded_digest,
                expected_digest=expected_digest,
            )
        )
    else:
        checks.append(_check("probe-integrity", "blocked", "probe_digest is missing"))

    if target_architecture and target_architecture == probe_architecture:
        checks.append(
            _check(
                "target-architecture",
                "pass",
                "manifest target architecture matches the probe",
                manifest_architecture=target_architecture,
                probe_architecture=probe_architecture,
            )
        )
    elif not target_architecture:
        checks.append(_check("target-architecture", "blocked", "manifest target.architecture is missing"))
    else:
        checks.append(
            _check(
                "target-architecture",
                "blocked",
                "manifest target architecture does not match the probe",
                manifest_architecture=target_architecture,
                probe_architecture=probe_architecture,
            )
        )

    if backend in advertised:
        checks.append(_check("backend-advertised", "pass", "backend is explicitly advertised by the target probe", backend=backend))
    elif backend in inferred:
        checks.append(
            _check(
                "backend-advertised",
                "blocked",
                "backend appears only in inferred claims; explicit advertisement is required",
                backend=backend,
            )
        )
    else:
        checks.append(_check("backend-advertised", "blocked", "backend is not advertised by the target probe", backend=backend))

    required_accelerators = _list_of_strings((manifest.get("backend_spec") or {}).get("accelerators"), "backend accelerators")
    missing_accelerators = sorted(set(required_accelerators) - set(accelerators))
    checks.append(
        _check(
            "backend-accelerator",
            "pass" if not missing_accelerators else "blocked",
            "required accelerator evidence is present" if not missing_accelerators else "required accelerator is absent from the probe",
            required=required_accelerators,
            available=accelerators,
            missing=missing_accelerators,
        )
    )

    runs = _normalise_runs(model_runs)
    matching_runs: list[dict[str, Any]] = []
    for run in runs:
        run_manifest = run.get("manifest") if isinstance(run.get("manifest"), dict) else {}
        run_backend = str(run.get("compiler_backend") or (run_manifest.get("compiler") or {}).get("backend") or "")
        run_architecture = str(run.get("target_architecture") or (run_manifest.get("target") or {}).get("architecture") or "")
        run_model = str(run.get("model_name") or (run_manifest.get("model") or {}).get("name") or "")
        run_dataset = str(run.get("dataset_name") or (run_manifest.get("dataset") or {}).get("name") or "")
        run_manifest_digest = str(run_manifest.get("manifest_digest") or run.get("manifest_digest") or "")
        if (
            run_manifest_digest == manifest["manifest_digest"]
            and run_backend == backend
            and run_architecture == target_architecture
            and run_model == str(manifest["model"]["name"])
            and run_dataset == str(manifest["dataset"]["name"])
        ):
            matching_runs.append(run)
    successful_runs = [
        run for run in matching_runs
        if str(run.get("task_status") or "") == "succeeded" and run.get("correctness") is True
    ]
    if successful_runs:
        checks.append(
            _check(
                "model-correctness",
                "pass",
                "matching model run passed correctness",
                matching_runs=len(matching_runs),
                successful_runs=len(successful_runs),
            )
        )
    elif matching_runs:
        checks.append(
            _check(
                "model-correctness",
                "blocked",
                "matching model runs exist but none passed correctness",
                matching_runs=len(matching_runs),
                successful_runs=0,
            )
        )
    else:
        checks.append(
            _check(
                "model-correctness",
                "blocked",
                "no successful correctness evidence matches this manifest and target",
                matching_runs=0,
                successful_runs=0,
            )
        )

    status = "ready" if all(item["status"] == "pass" for item in checks) else "blocked"
    return {
        "audit": AUDIT_VERSION,
        "status": status,
        "manifest_digest": manifest["manifest_digest"],
        "model": manifest["model"],
        "dataset": manifest["dataset"],
        "backend": backend,
        "target": target,
        "probe_digest": probe.get("probe_digest"),
        "checks": checks,
        "evidence": {
            "advertised_backends": advertised,
            "inferred_backends": inferred,
            "matching_model_runs": len(matching_runs),
            "successful_model_runs": len(successful_runs),
        },
        "scientific_conclusion_allowed": False,
        "interpretation": "deployment evidence gate only; no performance, causal, or scientific conclusion",
    }

