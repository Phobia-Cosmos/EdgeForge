#!/usr/bin/env python3
"""Plan or execute the versioned full ISRUC continuous-LoP experiment."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _inside_repository(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT.resolve())
        return True
    except ValueError:
        return False


def _ensure_role_view(plan: dict[str, Any], data_root: Path, view: Path) -> None:
    roles = plan.get("roles")
    if not isinstance(roles, dict):
        raise ValueError("plan is missing roles")
    role_sets: dict[str, set[int]] = {}
    for role in ("source", "target", "retention"):
        raw_subjects = roles.get(role)
        if not isinstance(raw_subjects, list) or not raw_subjects:
            raise ValueError(f"plan role {role!r} must be a non-empty list")
        subjects = [int(item) for item in raw_subjects]
        if len(subjects) != len(set(subjects)):
            raise ValueError(f"plan role {role!r} contains duplicate subjects")
        role_sets[role] = set(subjects)
    if any(role_sets[left] & role_sets[right] for left, right in (("source", "target"), ("source", "retention"), ("target", "retention"))):
        raise ValueError("source, target and retention subjects must be disjoint")
    view.mkdir(parents=True, exist_ok=True)
    for role in ("source", "target", "retention"):
        subjects = roles.get(role)
        if not isinstance(subjects, list) or not subjects:
            raise ValueError(f"plan role {role!r} must be a non-empty list")
        for raw_subject in subjects:
            subject = int(raw_subject)
            target = (data_root / str(subject)).resolve()
            if not (target / "data").is_dir() or not (target / "label").is_dir():
                raise FileNotFoundError(f"missing canonical subject payload: {target}")
            link = view / role / str(subject)
            link.parent.mkdir(parents=True, exist_ok=True)
            # A phase may start concurrently with another phase.  Check the
            # link state repeatedly so the tiny create/metadata visibility
            # window cannot turn an identical symlink into a false conflict.
            for _attempt in range(20):
                if link.is_symlink():
                    resolved = link.resolve()
                    if resolved != target:
                        raise ValueError(f"existing role link points elsewhere: {link} -> {resolved}")
                    break
                if link.exists():
                    time.sleep(0.001)
                    continue
                try:
                    link.symlink_to(target, target_is_directory=True)
                except FileExistsError:
                    time.sleep(0.001)
                    continue
                break
            else:
                if link.is_symlink() and link.resolve() == target:
                    continue
                raise FileExistsError(f"role view entry is not an identical symlink: {link}")
    _write_json_atomic(view / "PLAN.json", plan)


def _command(
    *,
    runner: Path,
    view: Path,
    output: Path,
    source_checkpoint_root: Path,
    architecture: str,
    seeds: list[int],
    source_subjects: list[int],
    target_subjects: list[int],
    retention_subjects: list[int],
    phase: dict[str, Any],
    device: str,
) -> list[str]:
    command = [
        sys.executable,
        str(runner),
        "--data-root", str(view),
        "--output-root", str(output),
        "--source-checkpoint-root", str(source_checkpoint_root),
        "--architectures", architecture,
        "--seeds", *(str(item) for item in seeds),
        "--source-subjects", *(str(item) for item in source_subjects),
        "--target-subjects", *(str(item) for item in target_subjects),
        "--retention-subjects", *(str(item) for item in retention_subjects),
        "--budgets", *(str(item) for item in phase["budgets"]),
        "--epochs", str(phase["source_epochs"]),
        "--batch-size", str(phase["batch_size"]),
        "--lr", str(phase["source_lr"]),
        "--adapt-lr", str(phase["adapt_lr"]),
        "--source-eval-fraction", str(phase["source_eval_fraction"]),
        "--retention-max-samples", str(phase["retention_max_samples"]),
        "--input-scale", str(phase.get("input_scale", 1.0)),
        "--device", device,
        "--resume",
    ]
    if phase.get("freeze_batch_norm", False):
        command.append("--freeze-batch-norm")
    return command


def _run_logged(command: list[str], *, cwd: Path, stdout_path: Path, stderr_path: Path) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
        stdout.write(f"\n# started_unix={started}\n")
        stdout.flush()
        completed = subprocess.run(command, cwd=cwd, env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, stdout=stdout, stderr=stderr, text=True, check=False)
        stdout.write(f"# finished_unix={time.time()} exit_code={completed.returncode}\n")
    return int(completed.returncode)


def build_commands(
    plan: dict[str, Any],
    config: dict[str, Any],
    *,
    phase_name: str,
    view: Path,
    output_root: Path,
    device: str,
) -> list[dict[str, Any]]:
    phases = config.get("phases")
    if not isinstance(phases, dict) or phase_name not in phases or not isinstance(phases[phase_name], dict):
        raise ValueError(f"unknown phase {phase_name!r}")
    phase = phases[phase_name]
    roles = plan["roles"]
    orders = plan["orders"]
    records: list[dict[str, Any]] = []
    for order_name in phase["orders"]:
        if order_name not in orders:
            raise ValueError(f"plan has no target order {order_name!r}")
        target_subjects = [int(item) for item in orders[order_name]]
        limit = phase.get("target_limit")
        if limit is not None:
            target_subjects = target_subjects[: int(limit)]
        for architecture in phase["architectures"]:
            run_output = output_root / "runs" / phase_name / order_name / str(architecture)
            command = _command(
                runner=ROOT / "scripts" / "run-eeg-architecture-continuous-lop.py",
                view=view,
                output=run_output,
                source_checkpoint_root=output_root / "source-checkpoints",
                architecture=str(architecture),
                seeds=[int(item) for item in phase["seeds"]],
                source_subjects=[int(item) for item in roles["source"]],
                target_subjects=target_subjects,
                retention_subjects=[int(item) for item in roles["retention"]],
                phase=phase,
                device=device,
            )
            records.append({
                "phase": phase_name,
                "order": order_name,
                "architecture": str(architecture),
                "seeds": [int(item) for item in phase["seeds"]],
                "target_subjects": target_subjects,
                "output": str(run_output),
                "command": command,
                "status": "planned",
            })
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "eeg-lop-full-isruc-v0.19.0.json")
    parser.add_argument("--data-root", type=Path, required=True, help="canonical flat ISRUC subject root")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("calibration", "confirmatory", "data-composition"), default="calibration")
    parser.add_argument("--device", help="override phase device; normally cuda or cpu")
    parser.add_argument("--minimum-free-gib", type=float, help="override the config output-space execution gate")
    parser.add_argument("--execute", action="store_true", help="run commands; default is a non-training plan")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    if _inside_repository(output_root):
        parser.error("--output-root must be outside the Git checkout")
    plan = _load_object(args.plan.resolve())
    config = _load_object(args.config.resolve())
    if plan.get("schema") != "edgeforge.isruc-full-lop-plan.v1":
        parser.error("unsupported or missing plan schema")
    data_root = args.data_root.resolve()
    view = output_root / "role-view"
    _ensure_role_view(plan, data_root, view)
    default_device = "cuda" if args.phase != "calibration" else "cuda"
    device = str(args.device or default_device)
    commands = build_commands(plan, config, phase_name=args.phase, view=view, output_root=output_root, device=device)
    manifest: dict[str, Any] = {
        "schema": "edgeforge.eeg-lop-full-run-manifest.v1",
        "experiment_version": config.get("experiment_version"),
        "phase": args.phase,
        "plan": str(args.plan.resolve()),
        "plan_digest": plan.get("plan_digest"),
        "config": str(args.config.resolve()),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "device": device,
        "execute": bool(args.execute),
        "commands": commands,
        "scientific_conclusion_allowed": False,
    }
    manifest_path = output_root / f"manifest-{args.phase}.json"
    _write_json_atomic(manifest_path, manifest)
    if not args.execute:
        print(json.dumps({"status": "planned", "phase": args.phase, "commands": len(commands), "manifest": str(manifest_path), "execute": False}, sort_keys=True))
        return 0

    resources = config.get("resources") if isinstance(config.get("resources"), dict) else {}
    minimum_free_gib = float(args.minimum_free_gib if args.minimum_free_gib is not None else resources.get("minimum_output_free_gib", 8.0))
    free_gib = shutil.disk_usage(output_root).free / (1024 ** 3)
    manifest["resource_preflight"] = {"output_free_gib": free_gib, "minimum_output_free_gib": minimum_free_gib}
    _write_json_atomic(manifest_path, manifest)
    if free_gib < minimum_free_gib:
        print(json.dumps({"status": "blocked-low-disk", "output_free_gib": free_gib, "minimum_output_free_gib": minimum_free_gib, "manifest": str(manifest_path)}, sort_keys=True))
        return 3

    for index, record in enumerate(commands):
        record["status"] = "running"
        record["started_unix"] = time.time()
        _write_json_atomic(manifest_path, manifest)
        log_root = output_root / "logs" / args.phase / record["order"] / record["architecture"]
        code = _run_logged(record["command"], cwd=ROOT, stdout_path=log_root / "stdout.log", stderr_path=log_root / "stderr.log")
        record["exit_code"] = code
        record["finished_unix"] = time.time()
        record["status"] = "succeeded" if code == 0 else "failed"
        _write_json_atomic(manifest_path, manifest)
        if code != 0:
            print(json.dumps({"status": "failed", "command_index": index, "manifest": str(manifest_path), "exit_code": code}, sort_keys=True))
            return code
    print(json.dumps({"status": "succeeded", "phase": args.phase, "commands": len(commands), "manifest": str(manifest_path), "scientific_conclusion_allowed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
