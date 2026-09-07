#!/usr/bin/env python3
"""Create an immutable, digest-bound lock for a passed EEG calibration phase.

The lock is intentionally small.  It records the exact configuration, plan,
run manifest and phase-aware analysis used to authorize the next phase while
leaving raw EEG, checkpoints and per-cell summaries outside the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


SCHEMA = "edgeforge.eeg-lop-calibration-lock.v1"


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _phase(config: dict[str, Any], name: str) -> dict[str, Any]:
    phases = config.get("phases")
    value = phases.get(name) if isinstance(phases, dict) else None
    if not isinstance(value, dict):
        raise ValueError(f"config has no {name!r} phase")
    return value


def build_lock(
    *,
    config_path: Path,
    plan_path: Path,
    manifest_path: Path,
    analysis_path: Path,
) -> dict[str, Any]:
    config = _load_object(config_path)
    plan = _load_object(plan_path)
    manifest = _load_object(manifest_path)
    analysis = _load_object(analysis_path)
    if analysis.get("phase") != "calibration":
        raise ValueError("calibration lock requires analysis.phase=calibration")
    if analysis.get("status") != "candidate-evidence-ready" or not analysis.get("candidate_evidence_ready"):
        raise ValueError("calibration analysis is not candidate-evidence-ready")
    audit = analysis.get("audit")
    if not isinstance(audit, dict) or not all(
        bool(audit.get(key))
        for key in ("completeness_passed", "design_consistency_passed", "learning_adequacy_passed", "sample_size_passed", "all_expected_cells_valid")
    ):
        raise ValueError("calibration analysis has a failed audit gate")
    if manifest.get("phase") != "calibration":
        raise ValueError("calibration lock requires a calibration run manifest")
    commands = manifest.get("commands")
    if not isinstance(commands, list) or not commands or any(
        not isinstance(command, dict) or command.get("status") != "succeeded"
        for command in commands
    ):
        raise ValueError("calibration run manifest contains failed or missing commands")

    calibration = _phase(config, "calibration")
    confirmatory = _phase(config, "confirmatory")
    acceptance = analysis.get("acceptance") if isinstance(analysis.get("acceptance"), dict) else {}
    lock: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "locked",
        "scientific_conclusion_allowed": False,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment_version": config.get("experiment_version"),
        "phase": "calibration",
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "plan": {"path": str(plan_path), "sha256": _sha256(plan_path), "plan_digest": plan.get("plan_digest")},
        "manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
            "expected_command_count": len(commands),
            "succeeded_command_count": sum(command.get("status") == "succeeded" for command in commands),
        },
        "analysis": {
            "path": str(analysis_path),
            "sha256": _sha256(analysis_path),
            "status": analysis.get("status"),
            "candidate_evidence_ready": bool(analysis.get("candidate_evidence_ready")),
            "valid_cells": audit.get("valid_cell_count"),
            "expected_cells": audit.get("expected_cell_count"),
            "warning_issue_count": audit.get("warning_issue_count", 0),
            "blocking_issue_count": audit.get("blocking_issue_count", 0),
        },
        "calibration_design": {
            "architectures": [str(item) for item in calibration.get("architectures", [])],
            "seeds": [int(item) for item in calibration.get("seeds", [])],
            "orders": [str(item) for item in calibration.get("orders", [])],
            "target_limit": calibration.get("target_limit"),
            "source_epochs": int(calibration["source_epochs"]),
            "budgets": [int(item) for item in calibration["budgets"]],
            "batch_size": int(calibration["batch_size"]),
            "source_lr": float(calibration["source_lr"]),
            "adapt_lr": float(calibration["adapt_lr"]),
            "input_scale": float(calibration.get("input_scale", config.get("input_scaling", {}).get("input_scale", 1.0))),
            "freeze_batch_norm": bool(calibration.get("freeze_batch_norm", False)),
        },
        "formal_design_locked_for_next_phase": {
            "architectures": [str(item) for item in confirmatory.get("architectures", [])],
            "seeds": [int(item) for item in confirmatory.get("seeds", [])],
            "orders": [str(item) for item in confirmatory.get("orders", [])],
            "target_limit": confirmatory.get("target_limit"),
            "source_epochs": int(confirmatory["source_epochs"]),
            "budgets": [int(item) for item in confirmatory["budgets"]],
            "batch_size": int(confirmatory["batch_size"]),
            "source_lr": float(confirmatory["source_lr"]),
            "adapt_lr": float(confirmatory["adapt_lr"]),
            "input_scale": float(confirmatory.get("input_scale", config.get("input_scaling", {}).get("input_scale", 1.0))),
            "freeze_batch_norm": bool(confirmatory.get("freeze_batch_norm", False)),
        },
        "acceptance": acceptance,
        "fresh_seed_scheme": "sha256(namespace, architecture, model_seed, target_subject); independent of stage/order",
        "source_checkpoint_policy": "one content-and-training-config signed checkpoint per architecture×seed, shared across orders",
        "lock_scope": "protocol readiness only; scientific LoP conclusions remain disabled",
    }
    lock["lock_digest"] = _json_digest(lock)
    return lock


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        lock = build_lock(
            config_path=args.config.resolve(),
            plan_path=args.plan.resolve(),
            manifest_path=args.manifest.resolve(),
            analysis_path=args.analysis.resolve(),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(lock, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, output)
    print(json.dumps({"status": lock["status"], "lock": str(output), "lock_digest": lock["lock_digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
